"use client";

import React from "react";
import { History, Trash2, XCircle } from "lucide-react";
import {
  ResearchHistoryEntry,
  formatHistoryTime,
  historyRunLabel,
} from "@/lib/research-history";

const STATUS_TONE: Record<ResearchHistoryEntry["status"], string> = {
  answered: "text-emerald-700",
  abstained: "text-amber-700",
  error: "text-rose-700",
};

export interface ResearchHistoryPanelProps {
  entries: ResearchHistoryEntry[];
  /** id of the run currently on screen, if any. */
  activeId?: string | null;
  onOpen?: (entry: ResearchHistoryEntry) => void;
  onDelete?: (id: string) => void;
  onClear?: () => void;
  /** Shorten the heading copy when the panel sits inside another card. */
  compact?: boolean;
  className?: string;
}

export function ResearchHistoryPanel({
  entries,
  activeId = null,
  onOpen,
  onDelete,
  onClear,
  compact = false,
  className = "",
}: ResearchHistoryPanelProps) {
  if (entries.length === 0) return null;

  return (
    <section
      aria-label="Saved research from this browser"
      className={`overflow-hidden rounded-xl border border-ink-200 bg-white shadow-card ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-200 bg-ink-100/60 px-4 py-3">
        <h2 className="flex items-center gap-2 text-[13px] font-bold text-ink-900">
          <History className="h-4 w-4 text-ink-400" />
          Saved research
        </h2>
        {onClear && entries.length > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="flex items-center gap-1 text-[11px] font-semibold text-ink-500 transition-colors hover:text-rose-700"
            aria-label="Clear all saved research from this browser"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear saved
          </button>
        )}
      </div>

      {!compact && (
        <p className="border-b border-ink-100 px-4 py-2.5 text-[11px] leading-relaxed text-ink-500">
          Answers stay in this browser so you can return to earlier questions. They
          are not stored on the research backend and are never shared.
        </p>
      )}

      <ul className="divide-y divide-ink-100">
        {entries.map((entry) => {
          const isActive = entry.id === activeId;
          return (
            <li
              key={entry.id}
              className={`flex flex-wrap items-center gap-3 px-4 py-3 transition-colors ${
                isActive ? "bg-brand-50/50" : "hover:bg-ink-100/50"
              }`}
            >
              <div className="min-w-0 flex-1">
                <p className="line-clamp-2 text-[13px] font-medium leading-snug text-ink-900">
                  {entry.question ? `“${entry.question}”` : "Question text unavailable"}
                </p>
                <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
                  <span
                    className={`font-semibold ${STATUS_TONE[entry.status] ?? "text-ink-600"}`}
                  >
                    {historyRunLabel(entry)}
                  </span>
                  <span className="text-ink-300">•</span>
                  <span className="font-mono text-ink-500">
                    {formatHistoryTime(entry.completedAt)}
                  </span>
                  <span className="text-ink-300">•</span>
                  <span className="text-ink-500">
                    {entry.docIds && entry.docIds.length > 0
                      ? `${entry.docIds.length} paper${entry.docIds.length === 1 ? "" : "s"} selected`
                      : "whole corpus"}
                  </span>
                  {isActive && (
                    <span className="pt-chip border-brand-200 bg-brand-50 text-brand-700">
                      Showing
                    </span>
                  )}
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-1.5">
                {onOpen && (
                  <button
                    type="button"
                    onClick={() => onOpen(entry)}
                    className="pt-btn-secondary text-[11px]"
                    aria-label={`Open saved answer for ${entry.question || "this question"}`}
                  >
                    Open
                  </button>
                )}
                {onDelete && (
                  <button
                    type="button"
                    onClick={() => onDelete(entry.id)}
                    className="rounded-md p-1.5 text-ink-400 transition-colors hover:bg-rose-50 hover:text-rose-700"
                    aria-label={`Remove saved answer for ${entry.question || "this question"}`}
                  >
                    <XCircle className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
