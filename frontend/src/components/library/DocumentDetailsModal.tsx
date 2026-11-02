"use client";

import React, { useEffect, useState } from "react";
import {
  X,
  FileText,
  Copy,
  Check,
  AlertTriangle,
  CheckCircle2,
  Layers,
  FileCheck,
  Hash,
  ShieldCheck,
} from "lucide-react";
import { Document } from "@/lib/types";
import { documentDisplayTitle } from "@/lib/display";

export interface DocumentDetailsModalProps {
  document: Document | null;
  isOpen: boolean;
  onClose: () => void;
  onInspectChunks: (doc: Document) => void;
}

export function DocumentDetailsModal({
  document,
  isOpen,
  onClose,
  onInspectChunks,
}: DocumentDetailsModalProps) {
  const [copiedId, setCopiedId] = useState<boolean>(false);
  const [copiedSha, setCopiedSha] = useState<boolean>(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !document) return null;

  const handleCopyId = async () => {
    try {
      await navigator.clipboard.writeText(document.doc_id);
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    } catch {
      setCopiedId(false);
    }
  };

  const handleCopySha = async () => {
    try {
      await navigator.clipboard.writeText(document.sha256);
      setCopiedSha(true);
      setTimeout(() => setCopiedSha(false), 2000);
    } catch {
      setCopiedSha(false);
    }
  };

  const warnings = document.extraction_warnings || [];
  /**
   * Older parser versions phrased the figure warning as if whole pages were skipped.
   * The stored wording is left exactly as recorded; the note beside it keeps the
   * meaning honest.
   */
  const hasLegacyImageWarning = warnings.some((w) =>
    /contain image blocks that were not extracted/i.test(w)
  );

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto bg-ink-900/60 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="doc-details-title"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl max-w-2xl w-full shadow-2xl border border-ink-200 overflow-hidden text-left animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-5 border-b border-ink-100 flex items-start justify-between gap-4 bg-ink-50/50">
          <div className="min-w-0">
            <span className="text-[11px] font-bold uppercase tracking-wider text-brand-600 bg-brand-50 px-2 py-0.5 rounded border border-brand-100">
              Document Metadata & Diagnostics
            </span>
            <h2
              id="doc-details-title"
              className="text-lg font-bold text-ink-900 mt-1 leading-snug line-clamp-2"
            >
              {documentDisplayTitle(document)}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-ink-400 hover:text-ink-700 hover:bg-ink-100 rounded-lg transition-colors shrink-0"
            aria-label="Close document details"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-5 max-h-[75vh] overflow-y-auto">
          {/* Metadata Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Document ID */}
            <div className="p-3 rounded-lg bg-ink-50 border border-ink-200/80">
              <span className="text-xs font-semibold text-ink-500 uppercase tracking-wider">
                Document ID
              </span>
              <div className="mt-1 flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-ink-900">
                  {document.doc_id}
                </span>
                <button
                  type="button"
                  onClick={handleCopyId}
                  className="p-1 text-ink-400 hover:text-ink-700 rounded transition-colors"
                  title="Copy Document ID"
                  aria-label="Copy Document ID"
                >
                  {copiedId ? (
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            </div>

            {/* Original Filename */}
            <div className="p-3 rounded-lg bg-ink-50 border border-ink-200/80">
              <span className="text-xs font-semibold text-ink-500 uppercase tracking-wider">
                Original Filename
              </span>
              <p className="mt-1 text-xs font-medium text-ink-800 font-mono truncate" title={document.filename}>
                {document.filename}
              </p>
            </div>

            {/* Page Count */}
            <div className="p-3 rounded-lg bg-ink-50 border border-ink-200/80">
              <span className="text-xs font-semibold text-ink-500 uppercase tracking-wider">
                Page Count
              </span>
              <p className="mt-1 text-sm font-bold text-ink-900 flex items-center gap-1.5">
                <FileCheck className="w-4 h-4 text-brand-600" />
                <span>{document.page_count} pages</span>
              </p>
            </div>

            {/* Chunker Version */}
            <div className="p-3 rounded-lg bg-ink-50 border border-ink-200/80">
              <span className="text-xs font-semibold text-ink-500 uppercase tracking-wider">
                Chunker Version
              </span>
              <p className="mt-1 text-xs font-mono font-bold text-ink-800">
                {document.chunker_version || "default"}
              </p>
            </div>
          </div>

          {/* SHA-256 Hash */}
          <div className="p-3 rounded-lg bg-ink-50 border border-ink-200/80">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-ink-500 uppercase tracking-wider flex items-center gap-1">
                <Hash className="w-3.5 h-3.5 text-ink-400" />
                SHA-256 Digest (Content Integrity)
              </span>
              <button
                type="button"
                onClick={handleCopySha}
                className="flex items-center gap-1 text-[11px] font-medium text-ink-600 hover:text-ink-900 transition-colors"
                aria-label="Copy SHA-256 hash"
              >
                {copiedSha ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-600" />
                    <span className="text-emerald-700">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    <span>Copy Full SHA-256</span>
                  </>
                )}
              </button>
            </div>
            <p className="mt-1 text-xs font-mono text-ink-700 break-all bg-white p-2 rounded border border-ink-200 select-all">
              {document.sha256}
            </p>
          </div>

          {/* Extraction Warnings & Diagnostics */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-500 mb-2 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-ink-400" />
              Extraction Warnings & Diagnostics
            </h3>

            {warnings.length > 0 ? (
              <div className="space-y-2">
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900">
                  <div className="flex items-center gap-2 font-semibold text-amber-800 mb-1.5">
                    <AlertTriangle className="w-4 h-4 text-amber-600" />
                    <span>{warnings.length} Parser Warning{warnings.length === 1 ? "" : "s"} Recorded</span>
                  </div>
                  <ul className="list-disc list-inside space-y-1 font-mono text-[11px] text-amber-950">
                    {warnings.map((warn, i) => (
                      <li key={i} className="break-words">
                        {warn}
                      </li>
                    ))}
                  </ul>
                  {hasLegacyImageWarning && (
                    <p className="mt-2 text-[11px] leading-relaxed text-amber-900">
                      Those pages still contribute their body text, table contents and
                      figure captions to the index — what is not read is the graphic drawn
                      inside the image. This wording was recorded by the parser version in
                      use when the paper was first ingested; open the chunk explorer to see
                      every passage actually stored.
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg flex items-center gap-2 text-xs text-emerald-800">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                <span className="font-medium">
                  Clean extraction: No parser warnings or OCR anomalies reported for this document.
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 bg-ink-50 border-t border-ink-200 flex items-center justify-between">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium text-ink-700 hover:text-ink-900 hover:bg-ink-200/70 rounded-lg transition-colors"
          >
            Close
          </button>

          <button
            type="button"
            onClick={() => {
              onClose();
              onInspectChunks(document);
            }}
            className="px-4 py-2 text-xs font-semibold text-white bg-brand-600 hover:bg-brand-700 rounded-lg shadow-sm transition-colors flex items-center gap-1.5"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Inspect Chunks ({document.page_count} pages)</span>
          </button>
        </div>
      </div>
    </div>
  );
}
