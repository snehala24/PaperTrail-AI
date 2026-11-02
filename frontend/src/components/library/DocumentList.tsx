"use client";

import React, { useState, useMemo } from "react";
import {
  BookOpen,
  Search,
  RefreshCw,
  AlertCircle,
  CheckSquare,
  Square,
  X,
  FileCheck,
} from "lucide-react";
import { Document } from "@/lib/types";
import { documentDisplayTitle } from "@/lib/display";
import { DocumentCard } from "./DocumentCard";

export interface DocumentListProps {
  documents: Document[];
  isLoading: boolean;
  error: string | null;
  selectedDocIds: string[];
  onToggleSelect: (docId: string) => void;
  onSelectAll?: () => void;
  onClearSelection?: () => void;
  onRefresh: () => void;
  onViewDetails: (doc: Document) => void;
  onInspectChunks: (doc: Document) => void;
  /** doc_id -> stored chunk count, from the real chunk endpoint. Badge hidden when absent. */
  chunkCounts?: Record<string, number> | null;
  className?: string;
}

export function DocumentList({
  documents,
  isLoading,
  error,
  selectedDocIds,
  onToggleSelect,
  onSelectAll,
  onClearSelection,
  onRefresh,
  onViewDetails,
  onInspectChunks,
  chunkCounts,
  className = "",
}: DocumentListProps) {
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Filter documents by title, filename, or doc_id
  const filteredDocuments = useMemo(() => {
    if (!searchQuery.trim()) return documents;
    const q = searchQuery.toLowerCase();
    return documents.filter(
      (doc) =>
        (documentDisplayTitle(doc).toLowerCase().includes(q)) ||
        (doc.filename && doc.filename.toLowerCase().includes(q)) ||
        (doc.doc_id && doc.doc_id.toLowerCase().includes(q))
    );
  }, [documents, searchQuery]);

  const totalPages = useMemo(() => {
    return documents.reduce((acc, doc) => acc + (doc.page_count || 0), 0);
  }, [documents]);

  const totalChunks = useMemo(() => {
    if (!chunkCounts) return null;
    return documents.reduce((acc, doc) => acc + (chunkCounts[doc.doc_id] || 0), 0);
  }, [documents, chunkCounts]);

  const allFilteredSelected =
    filteredDocuments.length > 0 &&
    filteredDocuments.every((doc) => selectedDocIds.includes(doc.doc_id));

  return (
    <div className={`space-y-4 ${className}`}>
      {/* List Toolbar */}
      <div className="flex flex-col justify-between gap-3 rounded-xl border border-ink-200 bg-white p-4 shadow-card sm:flex-row sm:items-center">
        <div className="flex items-center space-x-2">
          <BookOpen className="w-5 h-5 text-brand-600" />
          <div>
            <h2 className="flex items-center gap-2 text-sm font-bold tracking-tight text-ink-900">
              <span>Indexed papers</span>
              <span className="rounded-full bg-ink-100 px-2 py-0.5 text-xs font-semibold text-ink-700">
                {documents.length} {documents.length === 1 ? "paper" : "papers"}
              </span>
            </h2>
            <p className="text-xs text-ink-500">
              {totalPages} total extracted pages across corpus
              {totalChunks !== null &&
                ` • ${totalChunks} indexed ${totalChunks === 1 ? "chunk" : "chunks"}`}
            </p>
          </div>
        </div>

        {/* Search & Actions */}
        <div className="flex items-center gap-2">
          <div className="relative min-w-[200px] sm:min-w-[240px]">
            <Search className="w-4 h-4 text-ink-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by title, file, or ID..."
              className="w-full pl-9 pr-7 py-1.5 text-xs bg-ink-50 border border-ink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500"
              aria-label="Filter documents by query"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-2 text-ink-400 hover:text-ink-600"
                aria-label="Clear document filter"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <button
            type="button"
            onClick={onRefresh}
            disabled={isLoading}
            className="p-2 text-ink-500 hover:text-ink-800 hover:bg-ink-100 rounded-lg border border-ink-200 transition-colors"
            title="Refresh documents list"
            aria-label="Refresh document library"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin text-brand-600" : ""}`} />
          </button>
        </div>
      </div>

      {/* Selection summary */}
      {documents.length > 0 && (
        <div className="flex items-center justify-between px-2 text-xs text-ink-600">
          <div className="flex items-center gap-3">
            {onSelectAll && onClearSelection && (
              <button
                type="button"
                onClick={allFilteredSelected ? onClearSelection : onSelectAll}
                className="inline-flex items-center gap-1.5 font-medium transition-colors hover:text-ink-900"
              >
                {allFilteredSelected ? (
                  <>
                    <CheckSquare className="w-4 h-4 text-brand-600" />
                    <span>Deselect All</span>
                  </>
                ) : (
                  <>
                    <Square className="w-4 h-4 text-ink-400" />
                    <span>Select All</span>
                  </>
                )}
              </button>
            )}
            <span className="text-ink-500">
              {selectedDocIds.length} of {documents.length}{" "}
              {selectedDocIds.length === 1 ? "paper" : "papers"} selected for the
              Assistant
            </span>
          </div>
        </div>
      )}

      {/* Error state — a failed request is never presented as an empty library. */}
      {error && (
        <div
          role="alert"
          className="flex items-start justify-between gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700"
        >
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-rose-600" />
            <div>
              <p className="text-sm font-semibold text-rose-900">
                The library could not be loaded
              </p>
              <p className="mt-0.5 break-words">{error}</p>
              <p className="mt-1 text-rose-700">
                Your papers are not lost — this is a connection problem.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onRefresh}
            className="shrink-0 rounded-md border border-rose-300 bg-white px-3 py-1 font-medium text-rose-700 transition-colors hover:bg-rose-50"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading skeleton */}
      {isLoading && documents.length === 0 && (
        <div className="space-y-3">
          {[1, 2].map((i) => (
            <div
              key={i}
              className="animate-pulse space-y-3 rounded-xl border border-ink-200 bg-white p-5 shadow-card"
              aria-hidden="true"
            >
              <div className="h-4 w-2/3 rounded bg-ink-200"></div>
              <div className="h-3 w-1/3 rounded bg-ink-100"></div>
              <div className="flex gap-2 pt-2">
                <div className="h-6 w-20 rounded bg-ink-100"></div>
                <div className="h-6 w-24 rounded bg-ink-100"></div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Genuinely empty corpus */}
      {!isLoading && !error && documents.length === 0 && (
        <div className="space-y-2 rounded-xl border border-ink-200 bg-white p-8 text-center shadow-card">
          <BookOpen className="mx-auto w-10 h-10 text-ink-300" />
          <h3 className="text-base font-bold text-ink-800">
            No Research Papers Ingested
          </h3>
          <p className="mx-auto max-w-md text-xs leading-relaxed text-ink-500">
            This workspace has no papers yet. Upload a PDF above to parse it, index
            its passages, and make it citable in the Research Assistant.
          </p>
        </div>
      )}

      {/* Search returned nothing */}
      {!isLoading && documents.length > 0 && filteredDocuments.length === 0 && (
        <div className="space-y-2 rounded-xl border border-ink-200 bg-white p-8 text-center shadow-card">
          <p className="text-sm font-semibold text-ink-800">No matching papers</p>
          <p className="text-xs text-ink-500">
            Nothing matches &ldquo;{searchQuery}&rdquo;. Try part of a title, the
            filename, or the document ID.
          </p>
          <button
            type="button"
            onClick={() => setSearchQuery("")}
            className="mt-2 text-xs font-semibold text-brand-600 hover:text-brand-800"
          >
            Clear filter
          </button>
        </div>
      )}

      {/* Documents Grid / List */}
      <div className="grid grid-cols-1 gap-4">
        {filteredDocuments.map((doc) => (
          <DocumentCard
            key={doc.doc_id}
            document={doc}
            isSelected={selectedDocIds.includes(doc.doc_id)}
            chunkCount={chunkCounts?.[doc.doc_id]}
            onToggleSelect={onToggleSelect}
            onViewDetails={onViewDetails}
            onInspectChunks={onInspectChunks}
          />
        ))}
      </div>
    </div>
  );
}
