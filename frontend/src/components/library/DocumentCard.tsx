"use client";

import React, { useState } from "react";
import {
  FileText,
  Copy,
  Check,
  AlertTriangle,
  CheckCircle2,
  Layers,
  FileCheck,
  ExternalLink,
  ChevronRight,
} from "lucide-react";
import { Document } from "@/lib/types";
import { documentDisplayTitle } from "@/lib/display";

export interface DocumentCardProps {
  document: Document;
  chunkCount?: number;
  isSelected?: boolean;
  onToggleSelect?: (docId: string) => void;
  onViewDetails: (doc: Document) => void;
  onInspectChunks: (doc: Document) => void;
  className?: string;
}

export function DocumentCard({
  document,
  chunkCount,
  isSelected = false,
  onToggleSelect,
  onViewDetails,
  onInspectChunks,
  className = "",
}: DocumentCardProps) {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopyId = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(document.doc_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback if clipboard API is restricted
      setCopied(false);
    }
  };

  const warningCount = document.extraction_warnings ? document.extraction_warnings.length : 0;
  const displayTitle = documentDisplayTitle(document);

  return (
    <article
      className={`bg-white rounded-xl border transition-all duration-150 p-5 shadow-sm hover:shadow-md ${
        isSelected
          ? "border-brand-500 ring-2 ring-brand-100 bg-brand-50/20"
          : "border-ink-200/90 hover:border-ink-300"
      } ${className}`}
      aria-label={`Document ${displayTitle}`}
    >
      <div className="flex items-start justify-between gap-3">
        {/* Selection checkbox + Paper Title & Filename */}
        <div className="flex items-start gap-3 min-w-0">
          {onToggleSelect && (
            <div className="pt-0.5 shrink-0">
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => onToggleSelect(document.doc_id)}
                aria-label={`Select ${displayTitle} for analysis`}
                className="w-4 h-4 text-brand-600 rounded border-ink-300 focus:ring-brand-500 cursor-pointer"
              />
            </div>
          )}

          <div className="min-w-0">
            <h3 className="text-base font-bold text-ink-900 tracking-tight leading-snug line-clamp-2 hover:text-brand-700 transition-colors">
              {displayTitle}
            </h3>
            <p className="text-xs text-ink-500 font-mono mt-0.5 flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-ink-400 shrink-0" />
              <span className="truncate">{document.filename}</span>
            </p>
          </div>
        </div>

        {/* doc_id pill with copy action */}
        <div className="shrink-0 flex items-center">
          <button
            type="button"
            onClick={handleCopyId}
            title="Copy doc_id to clipboard"
            aria-label={`Copy document ID ${document.doc_id}`}
            className="flex items-center gap-1 px-2 py-0.5 text-xs font-mono rounded bg-ink-100 hover:bg-ink-200 text-ink-700 border border-ink-200 transition-colors"
          >
            <span>{document.doc_id}</span>
            {copied ? (
              <Check className="w-3 h-3 text-emerald-600" />
            ) : (
              <Copy className="w-3 h-3 text-ink-400 hover:text-ink-600" />
            )}
          </button>
        </div>
      </div>

      {/* Metadata Badges */}
      <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
        {/* Page Count */}
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-ink-100 text-ink-700 font-medium">
          <FileCheck className="w-3.5 h-3.5 text-ink-500" />
          <span>{document.page_count} {document.page_count === 1 ? "page" : "pages"}</span>
        </span>

        {/* Chunk Count if available */}
        {typeof chunkCount === "number" && (
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-brand-50 text-brand-700 font-medium border border-brand-100">
            <Layers className="w-3.5 h-3.5 text-brand-500" />
            <span>{chunkCount} chunks</span>
          </span>
        )}

        {/* Chunker Version */}
        {document.chunker_version && (
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono bg-ink-50 text-ink-600 border border-ink-200">
            {document.chunker_version}
          </span>
        )}

        {/* Extraction Warnings Status */}
        {warningCount > 0 ? (
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-50 text-amber-800 border border-amber-200/80 font-medium text-[11px]"
            title={document.extraction_warnings.join("\n")}
          >
            <AlertTriangle className="w-3 h-3 text-amber-600 shrink-0" />
            <span>{warningCount} {warningCount === 1 ? "warning" : "warnings"}</span>
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-800 border border-emerald-200/70 font-medium text-[11px]">
            <CheckCircle2 className="w-3 h-3 text-emerald-600 shrink-0" />
            <span>Clean parse</span>
          </span>
        )}

        {/* SHA-256 fingerprint */}
        <span className="text-[11px] text-ink-400 font-mono ml-auto truncate hidden sm:inline" title={document.sha256}>
          sha256: {document.sha256.substring(0, 10)}…
        </span>
      </div>

      {/* Card Footer Actions */}
      <div className="mt-4 pt-3 border-t border-ink-100 flex items-center justify-between">
        <button
          type="button"
          onClick={() => onViewDetails(document)}
          className="text-xs font-semibold text-ink-600 hover:text-ink-900 flex items-center gap-1 py-1 px-2 rounded hover:bg-ink-100 transition-colors"
          aria-label={`View metadata details for ${displayTitle}`}
        >
          <span>Metadata & Diagnostics</span>
          <ChevronRight className="w-3.5 h-3.5 text-ink-400" />
        </button>

        <button
          type="button"
          onClick={() => onInspectChunks(document)}
          className="text-xs font-semibold text-brand-600 hover:text-brand-800 bg-brand-50 hover:bg-brand-100/80 border border-brand-200/60 px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5"
          aria-label={`Inspect chunks for ${displayTitle}`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Inspect Chunks</span>
        </button>
      </div>
    </article>
  );
}
