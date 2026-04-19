"""Hybrid retrieval: BM25 + FAISS MMR, then cross-encoder rerank."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


def _doc_key(doc: Document) -> str:
    h = hashlib.sha256(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:24]
    src = (doc.metadata or {}).get("source", "")
    return f"{h}:{src}"


def split_pdf_to_chunks(pdf_path: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return chunks


@dataclass
class RetrievalIndex:
    chunks: list[Document]
    vector_store: FAISS
    bm25: BM25Retriever
    cross_encoder: CrossEncoder
    embedding_model_name: str

    def retrieve(self, question: str, *, k_bm25: int = 12, k_mmr: int = 12, fetch_k: int = 48, lambda_mult: float = 0.55, k_final: int = 5) -> list[Document]:
        bm25 = self.bm25
        bm25.k = k_bm25

        mmr = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_mmr, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

        from_bm25 = bm25.invoke(question)
        from_mmr = mmr.invoke(question)

        merged: list[Document] = []
        seen: set[str] = set()
        for d in from_bm25 + from_mmr:
            key = _doc_key(d)
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)

        if not merged:
            return []

        pairs: list[list[str]] = []
        contents: list[str] = []
        for d in merged:
            text = d.page_content[:8000]
            contents.append(text)
            pairs.append([question, text])

        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, merged), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:k_final]]


def build_index(chunks: list[Document], embedding_model_name: str, reranker_name: str) -> RetrievalIndex:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    bm25 = BM25Retriever.from_documents(chunks)
    cross = CrossEncoder(reranker_name)
    return RetrievalIndex(
        chunks=chunks,
        vector_store=vector_store,
        bm25=bm25,
        cross_encoder=cross,
        embedding_model_name=embedding_model_name,
    )
