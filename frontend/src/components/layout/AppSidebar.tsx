"use client";

import React from "react";
import {
  LayoutDashboard,
  BookOpen,
  Sparkles,
  FileSearch,
  GitCompare,
  X,
  ShieldCheck,
  Library,
} from "lucide-react";
import { CorpusStats } from "@/lib/corpus";
import { documentDisplayTitle } from "@/lib/display";

export type WorkspaceSection =
  | "dashboard"
  | "library"
  | "assistant"
  | "evidence"
  | "comparisons";

export interface NavItemConfig {
  id: WorkspaceSection;
  label: string;
  shortDesc: string;
  icon: React.ComponentType<{ className?: string }>;
}

export const NAV_ITEMS: NavItemConfig[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    shortDesc: "Workspace at a glance",
    icon: LayoutDashboard,
  },
  {
    id: "library",
    label: "Research Library",
    shortDesc: "Papers & ingestion",
    icon: BookOpen,
  },
  {
    id: "assistant",
    label: "Research Assistant",
    shortDesc: "Grounded answers",
    icon: Sparkles,
  },
  {
    id: "evidence",
    label: "Evidence Explorer",
    shortDesc: "Claims & citations",
    icon: FileSearch,
  },
  {
    id: "comparisons",
    label: "Cross-Paper Analysis",
    shortDesc: "Agreement & conflict",
    icon: GitCompare,
  },
];

/**
 * Compact corpus label: the arXiv id derived from the stored filename when
 * present (e.g. "2610.32878v1.pdf" -> "arXiv:2610.32878v1"), else the doc id.
 */
export function corpusLabel(filename?: string | null, docId?: string): string {
  const stem = (filename || "").replace(/\.pdf$/i, "").trim();
  if (stem) return `arXiv:${stem}`;
  return docId || "unknown";
}

export interface AppSidebarProps {
  activeSection: WorkspaceSection;
  onSelectSection: (section: WorkspaceSection) => void;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
  /** Real corpus statistics derived from backend endpoints; null while loading or on failure. */
  corpus?: CorpusStats | null;
  /** Reason the corpus could not be read, if any. */
  corpusError?: string | null;
}

export function AppSidebar({
  activeSection,
  onSelectSection,
  isMobileOpen,
  onCloseMobile,
  corpus = null,
  corpusError = null,
}: AppSidebarProps) {
  return (
    <>
      {/* Mobile backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-ink-900/40 backdrop-blur-sm md:hidden"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex w-[264px] flex-col border-r border-ink-200 bg-white transition-transform duration-200 ease-out md:static md:translate-x-0 ${
          isMobileOpen ? "translate-x-0 shadow-lg" : "-translate-x-full"
        }`}
        aria-label="Workspace navigation"
      >
        {/* Brand */}
        <div className="flex items-center justify-between gap-2 px-4 py-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <div
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-900 text-[13px] font-bold tracking-tight text-white shadow-2xs"
              aria-hidden="true"
            >
              PT
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="truncate text-[13px] font-bold tracking-tight text-ink-900">
                  PaperTrail AI
                </span>
                <span className="shrink-0 rounded border border-ink-200 bg-ink-100 px-1 py-px text-[9px] font-semibold text-ink-500">
                  v0.1
                </span>
              </div>
              <p className="truncate text-[11px] text-ink-500">
                Research evidence workspace
              </p>
            </div>
          </div>

          <button
            onClick={onCloseMobile}
            className="rounded-md p-1.5 text-ink-400 hover:bg-ink-100 hover:text-ink-700 md:hidden"
            aria-label="Close navigation sidebar"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Primary navigation */}
        <nav className="flex-1 overflow-y-auto px-3 pb-2">
          <p className="px-2 pb-1.5 pt-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-400">
            Workspace
          </p>
          <ul className="space-y-0.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;

              return (
                <li key={item.id}>
                  <button
                    onClick={() => {
                      onSelectSection(item.id);
                      onCloseMobile();
                    }}
                    className={`group relative flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors duration-150 ${
                      isActive
                        ? "bg-brand-50 text-brand-800"
                        : "text-ink-600 hover:bg-ink-100 hover:text-ink-900"
                    }`}
                    aria-current={isActive ? "page" : undefined}
                  >
                    {isActive && (
                      <span
                        className="absolute -left-3 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-brand-600"
                        aria-hidden="true"
                      />
                    )}
                    <Icon
                      className={`h-4 w-4 shrink-0 transition-colors ${
                        isActive
                          ? "text-brand-600"
                          : "text-ink-400 group-hover:text-ink-600"
                      }`}
                    />
                    <span className="min-w-0 flex-1">
                      <span
                        className={`block truncate text-[13px] ${
                          isActive ? "font-semibold" : "font-medium"
                        }`}
                      >
                        {item.label}
                      </span>
                      <span
                        className={`block truncate text-[11px] ${
                          isActive ? "text-brand-700/75" : "text-ink-400"
                        }`}
                      >
                        {item.shortDesc}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Corpus panel — counts always derived from live backend data */}
        <div className="mx-3 mb-3 rounded-lg border border-ink-200 bg-ink-100/70 p-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-[11px] font-semibold text-ink-700">
              <Library className="h-3.5 w-3.5 text-brand-600" />
              Index
            </span>
            <span className="rounded border border-ink-200 bg-white px-1.5 py-px font-mono text-[10px] text-ink-600">
              {corpus
                ? `${corpus.documents.length} ${
                    corpus.documents.length === 1 ? "paper" : "papers"
                  }`
                : corpusError
                ? "Unavailable"
                : "Loading…"}
            </span>
          </div>

          {corpus && corpus.documents.length > 0 && (
            <ul className="max-h-36 space-y-1 overflow-y-auto pr-0.5 text-[11px]">
              {corpus.documents.map((doc) => (
                <li
                  key={doc.doc_id}
                  className="flex items-baseline justify-between gap-2"
                  title={documentDisplayTitle(doc)}
                >
                  <span className="truncate text-ink-600">
                    {corpusLabel(doc.filename, doc.doc_id)}
                  </span>
                  <span className="shrink-0 font-mono text-[10px] text-ink-500">
                    {doc.page_count}p
                    {typeof corpus.chunkCounts[doc.doc_id] === "number"
                      ? ` • ${corpus.chunkCounts[doc.doc_id]}c`
                      : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}

          {corpus && corpus.documents.length === 0 && (
            <p className="text-[11px] text-ink-500">No papers indexed yet.</p>
          )}

          {corpusError && (
            <p className="text-[11px] text-amber-700">
              Corpus stats unavailable — the backend could not be reached.
            </p>
          )}

          <div className="mt-2.5 flex items-center justify-between gap-2 border-t border-ink-200 pt-2">
            <span className="flex items-center gap-1 text-[10px] font-medium text-emerald-700">
              <ShieldCheck className="h-3 w-3 text-emerald-600" />
              Verify-or-abstain
            </span>
            <span className="font-mono text-[10px] text-ink-500">
              {corpus ? `${corpus.totalPages}p / ${corpus.totalChunks}c` : "—"}
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
