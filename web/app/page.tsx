"use client";

import { useCallback, useEffect, useState } from "react";
import { FlowTimeline } from "@/components/FlowTimeline";
import { UploadZone } from "@/components/UploadZone";
import { apiBase, createSession, healthCheck, runQuery, uploadPdf, type QueryResponse } from "@/lib/api";

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [indexed, setIndexed] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [queryBusy, setQueryBusy] = useState(false);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [chunkInfo, setChunkInfo] = useState<string | null>(null);
  const [pageOrigin, setPageOrigin] = useState("");

  useEffect(() => {
    setPageOrigin(typeof window !== "undefined" ? window.location.origin : "");
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const ok = await healthCheck();
      if (!cancelled) setApiOk(ok);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const ensureSession = useCallback(async () => {
    if (sessionId) return sessionId;
    const { session_id } = await createSession();
    setSessionId(session_id);
    return session_id;
  }, [sessionId]);

  const onUpload = useCallback(
    async (file: File) => {
      setError(null);
      setChunkInfo(null);
      setUploadBusy(true);
      try {
        const sid = await ensureSession();
        const r = await uploadPdf(sid, file);
        setIndexed(true);
        setChunkInfo(`${r.chunks} chunks indexed`);
        setResult(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setIndexed(false);
      } finally {
        setUploadBusy(false);
      }
    },
    [ensureSession],
  );

  const onAsk = useCallback(async () => {
    if (!sessionId || !indexed) return;
    const q = question.trim();
    if (!q) return;
    setError(null);
    setQueryBusy(true);
    try {
      const r = await runQuery(sessionId, q);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setQueryBusy(false);
    }
  }, [sessionId, indexed, question]);

  return (
    <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <header className="mb-10 border-b border-surface-border pb-8">
        <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">Corrective RAG</h1>
        <p className="mt-3 max-w-2xl text-slate-400">
          Upload a document, then ask a question. The pipeline retrieves with hybrid search and reranking, grades chunks, optionally hits the web, refines context, and answers.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-3 text-sm">
          <span className="rounded-md bg-surface-raised px-2 py-1 font-mono text-xs text-slate-400">API: {apiBase()}</span>
          {pageOrigin ? (
            <span className="rounded-md bg-surface-raised px-2 py-1 font-mono text-xs text-slate-500">App: {pageOrigin}</span>
          ) : null}
          {apiOk === false ? (
            <span className="text-rose-400">
              Backend unreachable — start API from <code className="text-rose-200">backend</code>:{" "}
              <code className="text-rose-200">uvicorn main:app --host 127.0.0.1 --port 8001 --reload</code>
            </span>
          ) : apiOk === true ? (
            <span className="text-emerald-400/90">Backend OK</span>
          ) : (
            <span className="text-slate-500">Checking backend…</span>
          )}
        </div>
      </header>

      <section className="grid gap-10 lg:grid-cols-2">
        <div className="space-y-6">
          <h2 className="text-lg font-medium text-slate-200">1. Document</h2>
          <UploadZone onFile={onUpload} busy={uploadBusy} disabled={queryBusy} />
          {sessionId ? (
            <p className="font-mono text-xs text-slate-500">
              session: <span className="text-slate-400">{sessionId}</span>
            </p>
          ) : null}
          {chunkInfo ? <p className="text-sm text-accent/90">{chunkInfo}</p> : null}

          <h2 className="pt-4 text-lg font-medium text-slate-200">2. Question</h2>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={4}
            disabled={!indexed || queryBusy}
            placeholder={indexed ? "Ask something about your PDF…" : "Upload a PDF first"}
            className="w-full resize-y rounded-lg border border-surface-border bg-surface-raised px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/30 disabled:opacity-50"
          />
          <button
            type="button"
            onClick={() => void onAsk()}
            disabled={!indexed || queryBusy || !question.trim()}
            className="rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-surface transition hover:bg-accent-dim disabled:cursor-not-allowed disabled:opacity-40"
          >
            {queryBusy ? "Running…" : "Run CRAG"}
          </button>
        </div>

        <div className="space-y-6">
          <h2 className="text-lg font-medium text-slate-200">Flow</h2>
          <div className="rounded-xl border border-surface-border bg-surface-raised/40 p-5 min-h-[320px]">
            {!result ? (
              <p className="text-sm text-slate-500">Run a query to see retrieve → evaluate → (web) → refine → generate.</p>
            ) : (
              <FlowTimeline events={result.events} />
            )}
          </div>

          {result ? (
            <div className="space-y-4 rounded-xl border border-surface-border bg-surface-raised/30 p-5">
              <div className="flex flex-wrap gap-2 text-xs">
                {result.verdict ? (
                  <span className="rounded bg-slate-800 px-2 py-1 font-mono text-amber-200/90">verdict: {result.verdict}</span>
                ) : null}
                {result.web_query ? (
                  <span className="rounded bg-slate-800 px-2 py-1 font-mono text-slate-300">web query: {result.web_query}</span>
                ) : null}
              </div>
              <h3 className="text-sm font-medium uppercase tracking-wide text-slate-500">Answer</h3>
              <p className="whitespace-pre-wrap text-slate-200 leading-relaxed">{result.answer}</p>
              {result.refined_context ? (
                <>
                  <h3 className="pt-2 text-sm font-medium uppercase tracking-wide text-slate-500">Refined context (preview)</h3>
                  <pre className="max-h-48 overflow-auto rounded-lg bg-surface p-3 font-mono text-xs text-slate-400 whitespace-pre-wrap">
                    {result.refined_context.slice(0, 4000)}
                    {result.refined_context.length > 4000 ? "\n…" : ""}
                  </pre>
                </>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>

      {error ? (
        <div className="fixed bottom-6 left-1/2 z-50 max-w-lg -translate-x-1/2 rounded-lg border border-rose-500/40 bg-rose-950/90 px-4 py-3 text-sm text-rose-100 shadow-lg">
          {error}
        </div>
      ) : null}
    </main>
  );
}
