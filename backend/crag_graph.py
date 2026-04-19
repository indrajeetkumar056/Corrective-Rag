"""CRAG LangGraph: retrieve -> score -> route -> optional web -> refine -> generate."""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from retrieval import RetrievalIndex


class DocEvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class KeepOrDrop(BaseModel):
    keep: bool


class WebQuery(BaseModel):
    query: str


class State(TypedDict):
    question: str
    docs: list[Document]
    good_docs: list[Document]
    verdict: str
    reason: str
    strips: list[str]
    kept_strips: list[str]
    refined_context: str
    web_query: str
    web_docs: list[Document]
    answer: str
    events: Annotated[list[dict[str, Any]], operator.add]


UPPER_TH = 0.7
LOWER_TH = 0.3


def _parse_json_model(text: str, model: type[BaseModel]) -> BaseModel:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no json")
    return model.model_validate(json.loads(match.group()))


def build_crag_app(index: RetrievalIndex, llm: ChatGroq, tavily_api_key: str | None):
    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a retrieval evaluator for RAG.\n"
                "Return ONLY JSON matching the schema.\n"
                "Scoring:\n"
                "- 0.8–1.0 → directly answers most of the question\n"
                "- 0.6–0.8 → highly relevant\n"
                "- 0.4–0.6 → partially relevant\n"
                "- 0.2–0.4 → weakly related\n"
                "- 0.0–0.2 → irrelevant\n"
                "If the chunk explains ANY important part of the question, score >= 0.4.",
            ),
            ("human", "Question: {question}\n\nChunk:\n{chunk}"),
        ]
    )
    doc_eval_text = doc_eval_prompt | llm
    try:
        doc_eval_structured = doc_eval_prompt | llm.with_structured_output(DocEvalScore)
    except Exception:
        doc_eval_structured = None

    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                'You are a strict relevance filter. Return ONLY JSON: {{"keep": true}} or {{"keep": false}}.',
            ),
            ("human", "Question: {question}\n\nSentence:\n{sentence}"),
        ]
    )
    filter_text = filter_prompt | llm
    try:
        filter_structured = filter_prompt | llm.with_structured_output(KeepOrDrop)
    except Exception:
        filter_structured = None

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user question into a short web search query (6–14 words). "
                'If recency matters, add (last 30 days). Return JSON: {{"query": "..."}}.',
            ),
            ("human", "Question: {question}"),
        ]
    )
    rewrite_text = rewrite_prompt | llm
    try:
        rewrite_structured = rewrite_prompt | llm.with_structured_output(WebQuery)
    except Exception:
        rewrite_structured = None

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer ONLY using the provided context. "
                "If context is empty or insufficient, say exactly: I don't know.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    tavily = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key) if tavily_api_key else None

    def _eval_doc(q: str, chunk: str) -> DocEvalScore:
        vars_ = {"question": q, "chunk": chunk}
        if doc_eval_structured is not None:
            try:
                out = doc_eval_structured.invoke(vars_)
                if isinstance(out, DocEvalScore):
                    return out
            except Exception:
                pass
        res = doc_eval_text.invoke(vars_)
        return _parse_json_model((res.content or "").strip(), DocEvalScore)

    def _filter_sentence(q: str, sentence: str) -> KeepOrDrop:
        vars_ = {"question": q, "sentence": sentence}
        if filter_structured is not None:
            try:
                out = filter_structured.invoke(vars_)
                if isinstance(out, KeepOrDrop):
                    return out
            except Exception:
                pass
        res = filter_text.invoke(vars_)
        return _parse_json_model((res.content or "").strip(), KeepOrDrop)

    def _rewrite(q: str) -> str:
        vars_ = {"question": q}
        if rewrite_structured is not None:
            try:
                out = rewrite_structured.invoke(vars_)
                if isinstance(out, WebQuery):
                    return out.query.strip()
            except Exception:
                pass
        res = rewrite_text.invoke(vars_)
        try:
            return _parse_json_model((res.content or "").strip(), WebQuery).query.strip()
        except Exception:
            return (res.content or q).strip()

    def retrieve_node(state: State) -> dict:
        q = state["question"]
        docs = index.retrieve(q)
        previews = [{"preview": d.page_content[:280], "metadata": dict(d.metadata or {})} for d in docs]
        return {
            "docs": docs,
            "events": [
                {
                    "step": "retrieve",
                    "label": "Retrieve (hybrid BM25 + MMR + rerank)",
                    "detail": f"{len(docs)} chunks selected",
                    "docs_preview": previews,
                }
            ],
        }

    def eval_each_doc_node(state: State) -> dict:
        q = state["question"]
        scores: list[float] = []
        good: list[Document] = []

        for d in state["docs"]:
            try:
                out = _eval_doc(q, d.page_content)
                score = float(out.score)
            except Exception:
                score = 0.0
            scores.append(score)
            if score > LOWER_TH:
                good.append(d)

        if any(s > UPPER_TH for s in scores):
            verdict, reason = "CORRECT", f"At least one chunk scored > {UPPER_TH}."
        elif len(scores) > 0 and all(s < LOWER_TH for s in scores):
            verdict, reason = "INCORRECT", f"All chunks scored < {LOWER_TH}."
        else:
            verdict, reason = "AMBIGUOUS", f"No chunk > {UPPER_TH}, not all < {LOWER_TH}."

        return {
            "good_docs": good,
            "verdict": verdict,
            "reason": reason,
            "events": [
                {
                    "step": "eval",
                    "label": "Evaluate retrieval",
                    "detail": reason,
                    "verdict": verdict,
                    "scores": scores,
                }
            ],
        }

    def decompose_to_sentences(text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def rewrite_query_node(state: State) -> dict:
        query = _rewrite(state["question"])
        return {
            "web_query": query,
            "events": [{"step": "rewrite", "label": "Rewrite query for web", "detail": query}],
        }

    def web_search_node(state: State) -> dict:
        if tavily is None:
            return {
                "web_docs": [],
                "events": [
                    {
                        "step": "web_search",
                        "label": "Web search",
                        "detail": "Skipped (no TAVILY_API_KEY)",
                    }
                ],
            }
        q = state.get("web_query") or state["question"]
        results = tavily.invoke({"query": q}) or []
        web_docs: list[Document] = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        return {
            "web_docs": web_docs,
            "events": [
                {
                    "step": "web_search",
                    "label": "Web search (Tavily)",
                    "detail": f"{len(web_docs)} results",
                    "urls": [r.get("url", "") for r in results[:5]],
                }
            ],
        }

    def route_after_eval(state: State) -> str:
        if state["verdict"] == "CORRECT":
            return "refine"
        if state["verdict"] == "AMBIGUOUS" and state.get("good_docs"):
            return "refine_ambiguous"
        return "rewrite_query"

    def refine(state: State) -> dict:
        q = state["question"]
        verdict = state.get("verdict")

        if verdict == "CORRECT":
            docs_to_use = state["good_docs"]
        elif verdict == "INCORRECT":
            docs_to_use = state["web_docs"]
        else:
            docs_to_use = list(state["good_docs"]) + list(state["web_docs"])

        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        strips = decompose_to_sentences(context)
        kept: list[str] = []

        for s in strips:
            try:
                out = _filter_sentence(q, s)
                keep = bool(out.keep)
            except Exception:
                keep = False
            if keep:
                kept.append(s)

        refined = "\n".join(kept).strip()
        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": refined,
            "events": [
                {
                    "step": "refine",
                    "label": "Refine context (sentence filter)",
                    "detail": f"{len(kept)} / {len(strips)} sentences kept",
                }
            ],
        }

    def generate(state: State) -> dict:
        out = (answer_prompt | llm).invoke({"question": state["question"], "context": state["refined_context"]})
        text = out.content or ""
        return {
            "answer": text,
            "events": [{"step": "generate", "label": "Generate answer", "detail": text[:400]}],
        }

    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("refine_ambiguous", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "refine_ambiguous": "refine_ambiguous",
            "rewrite_query": "rewrite_query",
        },
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("refine_ambiguous", "generate")
    g.add_edge("generate", END)

    return g.compile()


def initial_state(question: str) -> dict:
    return {
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
        "events": [{"step": "start", "label": "Question received", "detail": question}],
    }
