"use client";

import { useCallback, useId, useState } from "react";

type Props = {
  disabled?: boolean;
  busy?: boolean;
  onFile: (file: File) => void;
};

export function UploadZone({ disabled, busy, onFile }: Props) {
  const [drag, setDrag] = useState(false);
  const inputId = useId();

  const pick = useCallback(
    (files: FileList | null) => {
      const f = files?.[0];
      if (!f) return;
      const name = f.name.toLowerCase();
      const type = (f.type || "").toLowerCase();
      if (!name.endsWith(".pdf") && type !== "application/pdf") {
        alert("Please choose a PDF file.");
        return;
      }
      onFile(f);
    },
    [onFile],
  );

  const inactive = disabled || busy;

  return (
    <label
      htmlFor={inputId}
      onDragOver={(e) => {
        e.preventDefault();
        if (!inactive) setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        if (inactive) return;
        pick(e.dataTransfer.files);
      }}
      className={[
        "block cursor-pointer rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
        drag ? "border-accent bg-accent/5" : "border-surface-border bg-surface-raised/50",
        inactive ? "cursor-not-allowed opacity-50 pointer-events-none" : "hover:border-accent/50",
      ].join(" ")}
    >
      <input
        id={inputId}
        type="file"
        accept="application/pdf,.pdf"
        className="sr-only"
        disabled={inactive}
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
      <p className="text-lg font-medium text-slate-200">{busy ? "Indexing PDF…" : "Drop a PDF here or click to browse"}</p>
      <p className="mt-2 text-sm text-slate-500">Max 20 MB. Text is chunked and indexed (BM25 + vectors + reranker).</p>
    </label>
  );
}
