# Corrective RAG (CRAG)

End-to-end **Corrective Retrieval-Augmented Generation** demo: upload a PDF, ask questions, and watch a **LangGraph** pipeline retrieve, score, optionally search the web, refine context, and generate a grounded answer. Includes a **Next.js** UI and a **FastAPI** backend.

## What it does

1. **Hybrid retrieval** — BM25 keyword search + dense **FAISS** retrieval with **MMR**, merged and **reranked** with a **cross-encoder** (`sentence-transformers`).
2. **Retrieval grading** — Each chunk gets an LLM relevance score; the graph routes on **CORRECT** / **AMBIGUOUS** / **INCORRECT** style verdicts (threshold-based).
3. **Corrective path** — Weak or ambiguous retrieval triggers **query rewrite** and optional **Tavily** web search, then context is **sentence-filtered** before generation.
4. **Grounded answers** — The model is instructed to answer only from context and to say **“I don’t know”** when evidence is insufficient.

## Repository layout

| Path | Role |
|------|------|
| `backend/` | FastAPI app (`main.py`), CRAG graph (`crag_graph.py`), retrieval index (`retrieval.py`) |
| `web/` | Next.js 14 app (upload + Q&A + flow timeline) |
| `req.txt` | Python dependencies (install from repo root) |
| `scripts/run-dev.bat` | Windows helper: venv + API + Next dev |

## Prerequisites

- **Python 3.11+** (recommended) and a virtual environment  
- **Node.js 18+** and npm (for the web app)  
- **Groq** API key ([Groq Console](https://console.groq.com/))  
- **Tavily** API key (optional, for web search fallback) — [Tavily](https://tavily.com/)

First-time model downloads (embeddings + reranker) can take a few minutes and need disk space.

## Environment variables

Create a `.env` file in the **repository root** or in `backend/` (the API loads dotenv from the working directory). **Do not commit `.env`** or real keys.

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` or `GROQ_KEY` | Yes | Groq API key for `ChatGroq` |
| `TAVILY_API_KEY` or `TAVILY_KEY` | No | Enables Tavily web search when retrieval is weak |
| `GROQ_MODEL` | No | Default: `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL_NAME` | No | Default: `sentence-transformers/all-MiniLM-L6-v2` |
| `RERANKER_MODEL_NAME` | No | Default: `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `PORT` / `HOST` | No | API bind (defaults `8001` / `127.0.0.1`) |
| `CORS_ORIGINS` | No | Comma-separated origins; if unset, localhost with any port is allowed via regex |

**Frontend:** `NEXT_PUBLIC_API_URL` (e.g. `http://127.0.0.1:8001`) must match where FastAPI runs.

Example `.env` (placeholders only):

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## Python setup

From the repo root (adjust venv path if yours differs):

```bash
python -m venv corrective-rag
# Windows
corrective-rag\Scripts\activate
# macOS / Linux
source corrective-rag/bin/activate

pip install -r req.txt
```

Dependencies used by the CRAG API include FastAPI, LangGraph, LangChain, FAISS, sentence-transformers, rank-bm25, and others listed in `req.txt`.

## Run the backend

```bash
cd backend
# Ensure GROQ_API_KEY or GROQ_KEY is set in the environment or .env
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Health check: `GET http://127.0.0.1:8001/health`

## Run the frontend

```bash
cd web
npm install
set NEXT_PUBLIC_API_URL=http://127.0.0.1:8001
npm run dev
```

On Windows PowerShell, use `$env:NEXT_PUBLIC_API_URL="http://127.0.0.1:8001"` before `npm run dev`.

Open the app at [http://localhost:3000](http://localhost:3000) (Next.js may use **3001** if 3000 is busy).

## Windows: one-shot dev script

If your venv lives at `corrective-rag\Scripts\activate.bat` relative to the repo root:

```bat
scripts\run-dev.bat
```

This starts **uvicorn** in a separate window on port **8001** and **Next.js** in the current window with `NEXT_PUBLIC_API_URL` set.

## HTTP API (summary)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions` | Create a session; returns `session_id` |
| `POST` | `/sessions/{session_id}/upload` | Upload a **PDF** (max **20 MB**); builds index and CRAG app |
| `POST` | `/sessions/{session_id}/query` | JSON body `{ "question": "..." }`; returns answer, verdict, events, doc previews |
| `GET` | `/health` | Liveness |

Query responses include an `events` array (retrieve, eval, rewrite, web search, refine, generate) for UI or debugging.

## Limits and notes

- **PDF only** for uploads; text is extracted with PyPDF-based loading and chunked with overlap.
- In-memory **sessions** are not persisted across API restarts.
- Without a Tavily key, web search is skipped (the graph still runs; see event details).

## License

Add a `LICENSE` file if you plan to distribute this repository.
