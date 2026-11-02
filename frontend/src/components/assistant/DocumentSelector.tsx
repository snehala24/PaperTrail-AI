"use client";

import React, { useState } from "react";
import {
  FileText,
  Filter,
  ChevronDown,
  ChevronUp,
  Globe,
  X,
  AlertTriangle,
} from "lucide-react";
import { Document } from "@/lib/types";
import { documentDisplayTitle } from "@/lib/display";

export interface DocumentSelectorProps {
  documents: Document[];
  selectedDocIds: string[];
  onToggleDoc: (docId: string) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
  disabled?: boolean;
  /** The document list is still being fetched from the backend. */
  isLoading?: boolean;
  /** The document list could not be fetched; distinct from an empty corpus. */
  loadError?: string | null;
  className?: string;
}

export function DocumentSelector({
  documents,
  selectedDocIds,
  onToggleDoc,
  onSelectAll,
  onClearAll,
  disabled = false,
  isLoading = false,
  loadError = null,
  className = "",
}: DocumentSelectorProps) {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const hasFilter = selectedDocIds.length > 0;
  const scopeTitle = hasFilter
    ? `Filtered Corpus (${selectedDocIds.length} of ${documents.length} papers)`
    : "Entire Corpus (All Papers)";
  const scopeDetail = hasFilter
    ? "Answers are drawn only from the papers selected below."
    : "Answers are drawn from every paper in the index.";

  return (
    <div
      className={`overflow-hidden rounded-xl border border-ink-200 bg-white shadow-card ${className}`}
    >
      {/* Scope summary */}
      <div className="flex items-center justify-between gap-3 border-b border-ink-100 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <div
            className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${
              hasFilter
                ? "bg-brand-100 text-brand-700"
                : "bg-emerald-50 text-emerald-700"
            }`}
            aria-hidden="true"
          >
            {hasFilter ? (
              <Filter className="h-4 w-4" />
            ) : (
              <Globe className="h-4 w-4" />
            )}
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13px] font-semibold tracking-tight text-ink-900">
                {scopeTitle}
              </span>
              <span
                className={`pt-chip ${
                  hasFilter
                    ? "border-brand-200 bg-brand-50 text-brand-700"
                    : "border-emerald-200 bg-emerald-50 text-emerald-700"
                }`}
              >
                {hasFilter ? "Filter Applied" : "Unfiltered"}
              </span>
            </div>
            <p className="mt-0.5 truncate text-[11px] text-ink-500">
              {scopeDetail}
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          {hasFilter && (
            <button
              type="button"
              disabled={disabled}
              onClick={onClearAll}
              className="rounded-md px-2 py-1 text-[12px] font-medium text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800 disabled:opacity-50"
              title="Search across all papers again"
            >
              Reset to All
            </button>
          )}

          <button
            type="button"
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-[12px] font-medium text-ink-600 transition-colors hover:bg-ink-100 hover:text-ink-900"
            aria-expanded={isOpen}
            aria-label={
              isOpen ? "Collapse paper selection" : "Expand paper selection"
            }
          >
            <span>{isOpen ? "Hide" : "Choose papers"}</span>
            {isOpen ? (
              <ChevronUp className="h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Selected papers, shown as removable chips while the list is collapsed */}
      {!isOpen && hasFilter && (
        <div className="flex flex-wrap items-center gap-1.5 border-b border-ink-100 bg-ink-100/50 px-4 py-2.5">
          {selectedDocIds.map((docId) => {
            const doc = documents.find((d) => d.doc_id === docId);
            const label = doc ? documentDisplayTitle(doc) : docId;
            return (
              <span
                key={docId}
                className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-ink-200 bg-white px-2 py-1 text-[12px] font-medium text-ink-700 shadow-2xs"
              >
                <FileText className="h-3 w-3 shrink-0 text-ink-400" />
                <span className="max-w-[240px] truncate" title={label}>
                  {label}
                </span>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => onToggleDoc(docId)}
                  className="rounded p-0.5 text-ink-400 transition-colors hover:text-rose-600 disabled:opacity-50"
                  aria-label={`Remove ${label} from the search scope`}
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            );
          })}
        </div>
      )}

      {isOpen && (
        <div className="space-y-2.5 px-4 py-3.5">
          <div className="flex items-center justify-between border-b border-ink-100 pb-2 text-[12px] text-ink-500">
            <span>Select which papers to search:</span>
            <div className="flex gap-3">
              <button
                type="button"
                disabled={disabled || documents.length === 0}
                onClick={onSelectAll}
                className="font-medium text-brand-600 transition-colors hover:text-brand-800 disabled:opacity-50"
              >
                Select all ({documents.length})
              </button>
              <button
                type="button"
                disabled={disabled || selectedDocIds.length === 0}
                onClick={onClearAll}
                className="font-medium text-ink-500 transition-colors hover:text-ink-800 disabled:opacity-50"
              >
                Clear
              </button>
            </div>
          </div>

          {documents.length === 0 ? (
            loadError ? (
              <div
                role="status"
                className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-3 text-center text-[12px] text-amber-800"
              >
                <p className="flex items-center justify-center gap-1.5 font-semibold">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                  Could not load the paper list from the backend.
                </p>
                <p className="mt-1 break-words text-amber-700">{loadError}</p>
                <p className="mt-1 text-[11px] text-amber-700/80">
                  Choosing papers is unavailable; submitting a question will still
                  search the full index if the backend is reachable.
                </p>
              </div>
            ) : isLoading ? (
              <div className="space-y-2 py-1" aria-hidden="true">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="h-10 animate-pulse rounded-lg bg-ink-100"
                  />
                ))}
              </div>
            ) : (
              <div className="py-4 text-center text-[12px] text-ink-500">
                No papers in the index yet. Add a PDF from the Research Library.
              </div>
            )
          ) : (
            <ul className="max-h-56 space-y-1.5 overflow-y-auto pr-1">
              {documents.map((doc) => {
                const isSelected = selectedDocIds.includes(doc.doc_id);
                return (
                  <li key={doc.doc_id}>
                    <label
                      className={`flex cursor-pointer items-start gap-2.5 rounded-lg border p-2.5 text-[13px] transition-colors ${
                        isSelected
                          ? "border-brand-300 bg-brand-50/60 text-ink-900"
                          : "border-ink-200 text-ink-700 hover:bg-ink-100/70"
                      } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        disabled={disabled}
                        onChange={() => onToggleDoc(doc.doc_id)}
                        className="mt-0.5 h-4 w-4 cursor-pointer rounded border-ink-300 text-brand-600 focus:ring-brand-500"
                        aria-label={`Select paper ${documentDisplayTitle(doc)}`}
                      />
                      <span className="min-w-0 flex-1">
                        <span
                          className={`block truncate ${
                            isSelected ? "font-semibold" : "font-medium"
                          }`}
                        >
                          {documentDisplayTitle(doc)}
                        </span>
                        <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink-500">
                          <span className="truncate">{doc.filename}</span>
                          <span aria-hidden="true">•</span>
                          <span className="shrink-0">
                            {doc.page_count} pages
                          </span>
                        </span>
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
