"use client";

import type { FlowEvent } from "@/lib/api";

const STEP_ORDER: Record<string, number> = {
  start: 0,
  retrieve: 1,
  eval: 2,
  rewrite: 3,
  web_search: 4,
  refine: 5,
  generate: 6,
};

function sortEvents(events: FlowEvent[]): FlowEvent[] {
  return [...events].sort((a, b) => {
    const sa = STEP_ORDER[a.step ?? ""] ?? 99;
    const sb = STEP_ORDER[b.step ?? ""] ?? 99;
    return sa - sb;
  });
}

export function FlowTimeline({ events }: { events: FlowEvent[] }) {
  const sorted = sortEvents(events);

  return (
    <ol className="relative border-l border-surface-border pl-6 space-y-6">
      {sorted.map((ev, i) => (
        <li key={`${ev.step}-${i}`} className="ml-1">
          <span className="absolute -left-[5px] mt-1.5 h-2.5 w-2.5 rounded-full border border-accent/40 bg-accent/90" />
          <p className="text-xs font-mono uppercase tracking-wider text-accent-dim/90">{ev.step ?? "event"}</p>
          <p className="font-medium text-slate-100">{ev.label}</p>
          {ev.detail ? (
            <p className="mt-1 text-sm text-slate-400 whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
              {ev.detail}
            </p>
          ) : null}
          {ev.verdict ? (
            <p className="mt-2 inline-flex rounded-md border border-surface-border bg-surface-raised px-2 py-0.5 text-xs font-mono text-amber-200/90">
              verdict: {ev.verdict}
            </p>
          ) : null}
          {ev.scores && ev.scores.length > 0 ? (
            <p className="mt-1 font-mono text-xs text-slate-500">chunk scores: [{ev.scores.map((s) => s.toFixed(2)).join(", ")}]</p>
          ) : null}
          {ev.urls && ev.urls.length > 0 ? (
            <ul className="mt-2 list-disc pl-4 text-xs text-slate-500">
              {ev.urls.filter(Boolean).map((u) => (
                <li key={u} className="truncate">
                  <a href={u} className="text-accent/80 hover:underline" target="_blank" rel="noreferrer">
                    {u}
                  </a>
                </li>
              ))}
            </ul>
          ) : null}
        </li>
      ))}
    </ol>
  );
}
