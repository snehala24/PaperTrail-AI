"use client";

import React, { useEffect, useState, useMemo } from "react";
import {
  X,
  Layers,
  Search,
  Filter,
  FileText,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  RefreshCw,
  Hash,
} from "lucide-react";
import { Document, Chunk, ChunkKind } from "@/lib/types";
import { documentDisplayTitle } from "@/lib/display";
import { getDocumentChunks, ApiClientError } from "@/lib/api";

export interface ChunkExplorerModalProps {
  document: Document | null;
  isOpen: boolean;
  onClose: () => void;
}

const CHUNK_KIND_LABELS: Record<ChunkKind, { label: string; badgeClass: string }> = {
  text: { label: "text", badgeClass: "bg-blue-50 text-blue-700 border-blue-200" },
  table: { label: "table", badgeClass: "bg-purple-50 text-purple-700 border-purple-200" },
  figure_ref: { label: "figure_ref", badgeClass: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  figure_text: { label: "figure_text", badgeClass: "bg-teal-50 text-teal-700 border-teal-200" },
  footnote: { label: "footnote", badgeClass: "bg-amber-50 text-amber-700 border-amber-200" },
  references: { label: "references", badgeClass: "bg-rose-50 text-rose-700 border-rose-200" },
};

export function ChunkExplorerModal({
  document,
  isOpen,
  onClose,
}: ChunkExplorerModalProps) {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState<string>("");
  const [kindFilter, setKindFilter] = useState<string>("all");
  const [pageFilter, setPageFilter] = useState<string>("all");
  const [copiedChunkId, setCopiedChunkId] = useState<string | null>(null);

  // Keyboard shortcut: Escape closes modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  // Load chunks when document changes or modal opens
  const fetchChunks = async () => {
    if (!document) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await getDocumentChunks(document.doc_id);
      setChunks(data);
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        setError(err.detail);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to load document chunks.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && document) {
      // Mark loading immediately so the empty-state copy cannot flash
      // before fetchChunks() sets isLoading in the next tick.
      setIsLoading(true);
      fetchChunks();
    } else {
      setChunks([]);
      setError(null);
      setIsLoading(false);
      setSearchQuery("");
      setKindFilter("all");
      setPageFilter("all");
    }
  }, [isOpen, document?.doc_id]);

  const handleCopyChunkId = async (chunkId: string) => {
    try {
      await navigator.clipboard.writeText(chunkId);
      setCopiedChunkId(chunkId);
      setTimeout(() => setCopiedChunkId(null), 2000);
    } catch {
      setCopiedChunkId(null);
    }
  };

  // Extract unique pages from chunks for the filter dropdown
  const uniquePages = useMemo(() => {
    const pages = new Set<number>();
    chunks.forEach((c) => pages.add(c.page));
    return Array.from(pages).sort((a, b) => a - b);
  }, [chunks]);

  // Filter chunks based on search query, kind, and page
  const filteredChunks = useMemo(() => {
    return chunks.filter((chunk) => {
      // 1. Text filter
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        const matchesText = chunk.text.toLowerCase().includes(query);
        const matchesHeading = chunk.section_heading
          ? chunk.section_heading.toLowerCase().includes(query)
          : false;
        const matchesId = chunk.chunk_id.toLowerCase().includes(query);
        if (!matchesText && !matchesHeading && !matchesId) {
          return false;
        }
      }

      // 2. Kind filter
      if (kindFilter !== "all" && chunk.kind !== kindFilter) {
        return false;
      }

      // 3. Page filter
      if (pageFilter !== "all" && chunk.page !== Number(pageFilter)) {
        return false;
      }

      return true;
    });
  }, [chunks, searchQuery, kindFilter, pageFilter]);

  if (!isOpen || !document) return null;

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto bg-ink-900/60 backdrop-blur-xs flex items-center justify-center p-3 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="chunk-explorer-title"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl max-w-4xl w-full shadow-2xl border border-ink-200 overflow-hidden text-left flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-ink-200 flex items-start justify-between gap-4 bg-ink-50/70 shrink-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-brand-600 bg-brand-50 px-2 py-0.5 rounded border border-brand-100 flex items-center gap-1">
                <Layers className="w-3 h-3" />
                Granular Chunk Explorer
              </span>
              <span className="text-xs font-mono text-ink-500 bg-ink-100 px-2 py-0.5 rounded">
                {document.doc_id}
              </span>
            </div>
            <h2
              id="chunk-explorer-title"
              className="text-base sm:text-lg font-bold text-ink-900 mt-1 truncate"
              title={documentDisplayTitle(document)}
            >
              {documentDisplayTitle(document)}
            </h2>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-ink-400 hover:text-ink-700 hover:bg-ink-100 rounded-lg transition-colors shrink-0"
            aria-label="Close chunk explorer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Filter Controls Toolbar */}
        <div className="px-6 py-3 border-b border-ink-100 bg-white shrink-0">
          <div className="grid grid-cols-1 sm:grid-cols-12 gap-2.5">
            {/* Search Input */}
            <div className="sm:col-span-6 relative">
              <Search className="w-4 h-4 text-ink-400 absolute left-3 top-2.5" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search chunks by keywords or section..."
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-ink-50 border border-ink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500"
                aria-label="Search chunks text or section heading"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2.5 top-2.5 text-ink-400 hover:text-ink-600"
                  aria-label="Clear chunk search query"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Kind Filter */}
            <div className="sm:col-span-3">
              <select
                value={kindFilter}
                onChange={(e) => setKindFilter(e.target.value)}
                className="w-full py-1.5 px-2.5 text-xs bg-ink-50 border border-ink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 text-ink-700"
                aria-label="Filter by semantic kind"
              >
                <option value="all">All Chunk Kinds</option>
                <option value="text">text</option>
                <option value="table">table</option>
                <option value="figure_ref">figure_ref</option>
                <option value="figure_text">figure_text</option>
                <option value="footnote">footnote</option>
                <option value="references">references</option>
              </select>
            </div>

            {/* Page Filter */}
            <div className="sm:col-span-3">
              <select
                value={pageFilter}
                onChange={(e) => setPageFilter(e.target.value)}
                className="w-full py-1.5 px-2.5 text-xs bg-ink-50 border border-ink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 text-ink-700"
                aria-label="Filter by page number"
              >
                <option value="all">All Pages ({document.page_count})</option>
                {uniquePages.map((page) => (
                  <option key={page} value={page.toString()}>
                    Page {page}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Counts & Status bar */}
          <div className="mt-2.5 flex items-center justify-between text-xs text-ink-500">
            <div>
              {!isLoading && !error && (
                <span aria-label="Chunk count summary">
                  Showing <strong className="text-ink-900">{filteredChunks.length}</strong> of{" "}
                  <strong className="text-ink-900">{chunks.length}</strong> chunks
                </span>
              )}
            </div>

            <button
              type="button"
              onClick={fetchChunks}
              disabled={isLoading}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-ink-600 hover:text-ink-900"
              title="Refresh chunks from server"
            >
              <RefreshCw className={`w-3 h-3 ${isLoading ? "animate-spin" : ""}`} />
              <span>Refresh Chunks</span>
            </button>
          </div>
        </div>

        {/* Chunks List Body */}
        <div className="p-6 overflow-y-auto space-y-4 flex-1 bg-ink-50/40">
          {isLoading ? (
            <div className="py-12 flex flex-col items-center justify-center space-y-3 text-ink-500">
              <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
              <p className="text-sm font-medium">Retrieving granular chunks from backend...</p>
            </div>
          ) : error ? (
            <div
              role="alert"
              className="p-4 bg-red-50 border border-red-200 rounded-xl text-center space-y-2 text-xs text-red-700"
            >
              <AlertCircle className="w-6 h-6 text-red-600 mx-auto" />
              <p className="font-semibold text-sm">Failed to Load Chunks</p>
              <p>{error}</p>
              <button
                type="button"
                onClick={fetchChunks}
                className="mt-2 px-3 py-1 bg-white border border-red-300 rounded text-red-700 hover:bg-red-50 font-medium inline-flex items-center gap-1"
              >
                <RefreshCw className="w-3 h-3" />
                Retry
              </button>
            </div>
          ) : chunks.length === 0 ? (
            <div className="py-12 text-center text-ink-500 space-y-2">
              <Layers className="w-8 h-8 text-ink-400 mx-auto" />
              <p className="text-sm font-semibold text-ink-800">No Chunks Found</p>
              <p className="text-xs max-w-sm mx-auto">
                No chunks were returned for this document. It may have zero extractable text or is currently being re-indexed.
              </p>
            </div>
          ) : filteredChunks.length === 0 ? (
            <div className="py-12 text-center text-ink-500 space-y-2">
              <Filter className="w-8 h-8 text-ink-400 mx-auto" />
              <p className="text-sm font-semibold text-ink-800">No Matching Chunks</p>
              <p className="text-xs">
                No chunks matched query &ldquo;{searchQuery}&rdquo; and active filters.
              </p>
              <button
                type="button"
                onClick={() => {
                  setSearchQuery("");
                  setKindFilter("all");
                  setPageFilter("all");
                }}
                className="mt-2 px-3 py-1 text-xs font-medium text-brand-600 bg-brand-50 hover:bg-brand-100 rounded border border-brand-200"
              >
                Clear all filters
              </button>
            </div>
          ) : (
            filteredChunks.map((chunk) => {
              const kindConfig =
                CHUNK_KIND_LABELS[chunk.kind] || {
                  label: chunk.kind,
                  badgeClass: "bg-ink-50 text-ink-700 border-ink-200",
                };

              const wordCount = chunk.text.trim().split(/\s+/).filter(Boolean).length;

              return (
                <div
                  key={chunk.chunk_id}
                  className="bg-white rounded-xl border border-ink-200 p-4 shadow-xs space-y-3 hover:border-ink-300 transition-colors"
                >
                  {/* Chunk Header Bar */}
                  <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-ink-100">
                    <div className="flex items-center gap-2">
                      {/* Copyable Chunk ID */}
                      <button
                        type="button"
                        onClick={() => handleCopyChunkId(chunk.chunk_id)}
                        className="inline-flex items-center gap-1 font-mono text-xs font-semibold text-ink-800 bg-ink-100 hover:bg-ink-200 px-2 py-0.5 rounded transition-colors"
                        title="Copy chunk ID"
                        aria-label={`Copy chunk ID ${chunk.chunk_id}`}
                      >
                        <span>{chunk.chunk_id}</span>
                        {copiedChunkId === chunk.chunk_id ? (
                          <Check className="w-3 h-3 text-emerald-600" />
                        ) : (
                          <Copy className="w-3 h-3 text-ink-400" />
                        )}
                      </button>

                      {/* Page badge */}
                      <span className="px-2 py-0.5 text-[11px] font-medium rounded bg-ink-100 text-ink-700">
                        Page {chunk.page}
                      </span>

                      {/* Semantic Kind Badge */}
                      <span
                        className={`px-2 py-0.5 text-[11px] font-semibold rounded border ${kindConfig.badgeClass}`}
                      >
                        {kindConfig.label}
                      </span>
                    </div>

                    {/* Word / Char count */}
                    <span className="text-[11px] text-ink-400 font-medium">
                      {wordCount} words • {chunk.text.length} chars
                    </span>
                  </div>

                  {/* Section Heading if available */}
                  {chunk.section_heading && (
                    <div className="text-xs font-semibold text-ink-700 flex items-center gap-1.5">
                      <Hash className="w-3 h-3 text-brand-600 shrink-0" />
                      <span>{chunk.section_heading}</span>
                    </div>
                  )}

                  {/* Verbatim Chunk Text */}
                  <div className="p-3 bg-ink-50/80 rounded-lg border border-ink-100 text-xs text-ink-800 leading-relaxed font-sans whitespace-pre-wrap selection:bg-brand-100 selection:text-brand-900">
                    {chunk.text}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 bg-ink-50 border-t border-ink-200 flex items-center justify-between shrink-0">
          <span className="text-xs text-ink-500">
            Total chunks in corpus index: <strong>{chunks.length}</strong>
          </span>

          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium text-ink-700 hover:text-ink-900 hover:bg-ink-200/70 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
