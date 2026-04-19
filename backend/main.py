"""FastAPI server for CRAG: PDF upload per session + query with flow events."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from crag_graph import build_crag_app, initial_state
from langchain_groq import ChatGroq
from retrieval import RetrievalIndex, build_index, split_pdf_to_chunks

load_dotenv()


@dataclass
class SessionData:
    session_id: str
    index: RetrievalIndex | None = None
    app: object | None = None
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )


SESSIONS: dict[str, SessionData] = {}


def _cors_settings() -> tuple[list[str], str | None]:
    """Allow browser dev servers on common Next.js ports (3000, 3001, …)."""
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins, None
    # Default: any localhost / 127.0.0.1 origin with a port (Next may use 3001 if 3000 is busy)
    return [], r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


_cors_origins, _cors_regex = _cors_settings()

app = FastAPI(title="CRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def _get_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY or GROQ_KEY in environment.")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model, temperature=0, groq_api_key=key)


@app.post("/sessions")
def create_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = SessionData(session_id=sid)
    return {"session_id": sid}


@app.post("/sessions/{session_id}/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF too large (max 20MB).")

    sess = SESSIONS[session_id]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        chunks = split_pdf_to_chunks(path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
        idx = build_index(chunks, sess.embedding_model, sess.reranker_model)
        llm = _get_llm()
        tavily_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_KEY")
        crag = build_crag_app(idx, llm, tavily_key)
        sess.index = idx
        sess.app = crag
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return {"ok": True, "chunks": len(chunks), "embedding_model": sess.embedding_model, "reranker_model": sess.reranker_model}


@app.post("/sessions/{session_id}/query")
def query_session(session_id: str, body: QueryBody):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    sess = SESSIONS[session_id]
    if sess.app is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first.")

    state = initial_state(body.question.strip())
    out = sess.app.invoke(state)

    docs_out = [
        {"page_content": d.page_content[:2000], "metadata": dict(d.metadata or {})}
        for d in out.get("docs") or []
    ]
    return {
        "answer": out.get("answer", ""),
        "verdict": out.get("verdict", ""),
        "reason": out.get("reason", ""),
        "web_query": out.get("web_query", ""),
        "refined_context": out.get("refined_context", ""),
        "events": out.get("events") or [],
        "docs": docs_out,
        "scores": next((e.get("scores") for e in reversed(out.get("events") or []) if e.get("step") == "eval"), None),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host=os.getenv("HOST", "127.0.0.1"), port=port, reload=True)
