const DEFAULT_API = "http://localhost:8001";

export function apiBase(): string {
  const u = process.env.NEXT_PUBLIC_API_URL?.trim();
  return u && u.length > 0 ? u.replace(/\/$/, "") : DEFAULT_API;
}

function networkMessage(err: unknown, url: string): string {
  const base = apiBase();
  if (err instanceof TypeError && String(err.message).toLowerCase().includes("fetch")) {
    return (
      `Cannot reach ${url} (API base ${base}). Start the backend on port 8001 and confirm NEXT_PUBLIC_API_URL. ` +
      `CORS errors also look like network failures: restart the API after updating the backend so localhost:3001 (or your dev port) is allowed.`
    );
  }
  return err instanceof Error ? err.message : String(err);
}

export async function createSession(): Promise<{ session_id: string }> {
  const url = `${apiBase()}/sessions`;
  let res: Response;
  try {
    res = await fetch(url, { method: "POST" });
  } catch (e) {
    throw new Error(networkMessage(e, url));
  }
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function uploadPdf(sessionId: string, file: File): Promise<{ ok: boolean; chunks: number }> {
  const fd = new FormData();
  fd.append("file", file);
  const url = `${apiBase()}/sessions/${sessionId}/upload`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      body: fd,
    });
  } catch (e) {
    throw new Error(networkMessage(e, url));
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      const t = await res.text();
      if (t) msg = t;
    }
    throw new Error(msg);
  }
  return res.json();
}

export type FlowEvent = {
  step?: string;
  label?: string;
  detail?: string;
  verdict?: string;
  scores?: number[];
  docs_preview?: { preview?: string; metadata?: Record<string, unknown> }[];
  urls?: string[];
};

export type QueryResponse = {
  answer: string;
  verdict: string;
  reason: string;
  web_query: string;
  refined_context: string;
  events: FlowEvent[];
  docs: { page_content: string; metadata: Record<string, unknown> }[];
  scores: number[] | null;
};

export async function runQuery(sessionId: string, question: string): Promise<QueryResponse> {
  const url = `${apiBase()}/sessions/${sessionId}/query`;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (e) {
    throw new Error(networkMessage(e, url));
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      const t = await res.text();
      if (t) msg = t;
    }
    throw new Error(msg);
  }
  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${apiBase()}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
