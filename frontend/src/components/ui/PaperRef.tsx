"use client";

import React from "react";
import { FileText } from "lucide-react";
import { usePaperTitleResolver } from "@/lib/document-catalogue";

export interface PaperRefProps {
  /** Document identifier returned by the backend. */
  docId?: string | null;
  /** 1-indexed page number, when known. */
  page?: number | null;
  /** Chunk identifier, shown as compact provenance. */
  chunkId?: string | null;
  className?: string;
}

/**
 * Renders where a piece of evidence came from, using the paper title from the
 * live index. When the title cannot be resolved it shows the raw doc_id rather
 * than guessing — provenance is never softened.
 */
export function PaperRef({
  docId,
  page,
  chunkId,
  className = "",
}: PaperRefProps) {
  const resolveTitle = usePaperTitleResolver();
  const title = resolveTitle(docId);

  if (!docId && !chunkId) return null;

  return (
    <div className={`flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 ${className}`}>
      {docId && (
        <span
          className="flex min-w-0 items-center gap-1.5 text-[12px] font-semibold text-ink-800"
          title={title ? `${title} (${docId})` : docId}
        >
          <FileText className="h-3.5 w-3.5 shrink-0 text-ink-400" />
          <span className="truncate">{title ?? docId}</span>
        </span>
      )}

      {typeof page === "number" && (
        <span className="shrink-0 rounded border border-ink-200 bg-ink-100 px-1.5 py-px font-mono text-[11px] text-ink-600">
          p. {page}
        </span>
      )}

      {chunkId && (
        <span
          className="shrink-0 truncate font-mono text-[10px] text-ink-400"
          title={chunkId}
        >
          {chunkId}
        </span>
      )}
    </div>
  );
}
