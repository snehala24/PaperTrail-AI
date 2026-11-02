"use client";

import React, { useState, useMemo } from "react";
import {
  GitCompare,
  Info,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from "lucide-react";
import { AnswerResponse, ClaimComparison, EvidenceLink } from "@/lib/types";
import { PageHeader } from "@/components/ui/PageHeader";
import { PaperRef } from "@/components/ui/PaperRef";
import {
  COMPARISON_META,
  comparisonMeta,
} from "@/lib/comparison-meta";
import { ComparisonType } from "@/lib/types";

function SideEvidence({ links }: { links: EvidenceLink[] }) {
  if (links.length === 0) {
    return (
      <p className="mt-2 text-[11px] italic leading-snug text-ink-400">
        No quoted passage was captured in the answer for this claim.
      </p>
    );
  }
  return (
    <div className="mt-2 space-y-2">
      {links.map((link) => (
        <blockquote
          key={`${link.claim_id}-${link.chunk_id}`}
          className="rounded-r-md border-l-2 border-ink-300 bg-white px-2.5 py-2"
        >
          <p className="font-mono text-[10px] text-ink-500">
            p. {link.page ?? "—"} · {link.chunk_id}
          </p>
          <p className="mt-1 break-words text-[12px] leading-relaxed text-ink-800">
            {link.quote ? `“${link.quote}”` : "No verbatim quotation was recorded for this passage."}
          </p>
        </blockquote>
      ))}
    </div>
  );
}

function ComparisonCard({
  comparison,
  index,
  evidenceByClaim,
}: {
  comparison: ClaimComparison;
  index: number;
  evidenceByClaim: Record<string, EvidenceLink[]>;
}) {
  const [expanded, setExpanded] = useState(true);
  const meta = comparisonMeta(comparison.comparison_type);
  const Icon = meta.icon;

  return (
    <article
      aria-label={`Comparison ${index + 1}: ${meta.shortLabel}`}
      className={`overflow-hidden rounded-xl border border-ink-200 bg-white shadow-card`}
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-b border-ink-100 px-4 py-3 text-left transition-colors hover:bg-ink-100/60"
        aria-expanded={expanded}
      >
        <span className="flex min-w-0 flex-wrap items-center gap-2.5">
          <span className={`pt-chip ${meta.badge}`}>
            <Icon className="h-3.5 w-3.5" />
            {meta.label}
          </span>
          <span className="font-mono text-[11px] text-ink-500">
            {comparison.claim_id_a} <span aria-hidden="true">↔</span>{" "}
            {comparison.claim_id_b}
          </span>
        </span>
        {expanded ? (
          <ChevronUp className="h-4 w-4 shrink-0 text-ink-400" />
        ) : (
          <ChevronDown className="h-4 w-4 shrink-0 text-ink-400" />
        )}
      </button>

      {expanded && (
        <div className="space-y-3.5 px-4 py-4">
          <p className="text-[12px] leading-relaxed text-ink-600">{meta.meaning}</p>

          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <div className="rounded-lg border border-ink-200 bg-ink-100/40 px-3 py-2.5">
              <p className="pt-eyebrow mb-1.5">Side A</p>
              <p className="mb-1 font-mono text-[11px] font-semibold text-ink-700">
                {comparison.claim_id_a}
              </p>
              <PaperRef docId={comparison.doc_id_a} />
              <SideEvidence links={evidenceByClaim[comparison.claim_id_a] ?? []} />
            </div>
            <div className="rounded-lg border border-ink-200 bg-ink-100/40 px-3 py-2.5">
              <p className="pt-eyebrow mb-1.5">Side B</p>
              <p className="mb-1 font-mono text-[11px] font-semibold text-ink-700">
                {comparison.claim_id_b}
              </p>
              <PaperRef docId={comparison.doc_id_b} />
              <SideEvidence links={evidenceByClaim[comparison.claim_id_b] ?? []} />
            </div>
          </div>

          {comparison.summary && (
            <div className="rounded-lg border border-ink-200 bg-white px-3.5 py-3">
              <p className="pt-eyebrow mb-1.5">What the evidence shows</p>
              <p className="text-[13px] leading-relaxed text-ink-800">
                {comparison.summary}
              </p>
            </div>
          )}

          {comparison.conflict_reason && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
              <div>
                <p className="pt-eyebrow mb-1 text-amber-700">
                  Source of the difference
                </p>
                <p className="text-[12px] leading-relaxed text-amber-900">
                  {comparison.conflict_reason}
                </p>
              </div>
            </div>
          )}

          {comparison.comparison_type === "unresolved" && (
            <div className="flex items-start gap-2 rounded-lg border border-ink-200 bg-ink-100/50 px-3 py-2.5 text-[12px] text-ink-600">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-400" />
              <span>
                Left unresolved deliberately: the retrieved evidence was not enough to
                call this agreement or contradiction.
              </span>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function NoAnswerState({ onGoToAssistant }: { onGoToAssistant?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center space-y-3.5 py-20 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-ink-200 bg-ink-100 text-ink-400">
        <GitCompare className="h-6 w-6" />
      </div>
      <div className="space-y-1">
        <h2 className="text-[15px] font-semibold text-ink-800">
          No comparisons yet
        </h2>
        <p className="mx-auto max-w-sm text-[13px] leading-relaxed text-ink-500">
          Ask a question that spans two or more papers. Where their findings meet,
          this view separates genuine contradictions from differences of method.
        </p>
      </div>
      {onGoToAssistant && (
        <button type="button" onClick={onGoToAssistant} className="pt-btn-primary">
          <Sparkles className="h-3.5 w-3.5" />
          Ask the Research Assistant
        </button>
      )}
    </div>
  );
}

function NoComparisonsState({ isAbstained }: { isAbstained: boolean }) {
  if (isAbstained) {
    return (
      <div
        role="alert"
        className="space-y-2 rounded-xl border border-amber-200 bg-amber-50 p-6 text-center"
      >
        <AlertTriangle className="mx-auto h-6 w-6 text-amber-500" />
        <h3 className="text-sm font-bold text-amber-950">
          Nothing to compare
        </h3>
        <p className="mx-auto max-w-md text-[13px] leading-relaxed text-amber-800">
          The system abstained from answering, so no cross-paper conclusions were
          drawn. Comparisons are never generated from unverified claims.
        </p>
      </div>
    );
  }
  return (
    <div className="flex flex-col items-center justify-center space-y-3 py-16 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-ink-200 bg-ink-100 text-ink-400">
        <GitCompare className="h-5 w-5" />
      </div>
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-ink-700">
          This answer did not span multiple papers
        </h3>
        <p className="mx-auto max-w-md text-[13px] leading-relaxed text-ink-500">
          Comparisons appear when verified evidence comes from two or more papers. Ask
          a cross-paper question, or widen the paper selection in the Assistant.
        </p>
      </div>
    </div>
  );
}

export interface CrossPaperComparisonsViewProps {
  answerResponse: AnswerResponse | null;
  onGoToAssistant?: () => void;
  /** Provenance line shown instead of the "current session" default for reopened runs. */
  sourceNote?: string | null;
}

export function CrossPaperComparisonsView({
  answerResponse,
  onGoToAssistant,
  sourceNote = null,
}: CrossPaperComparisonsViewProps) {
  const [filterType, setFilterType] = useState<string>("all");
  const [filterDocId, setFilterDocId] = useState<string>("all");

  const comparisons = answerResponse?.comparisons ?? [];
  const isAbstained = answerResponse?.status === "abstained";
  const isError = answerResponse?.status === "error";

  /** Exact quoted passages behind each claim, so both sides of a finding are inspectable. */
  const evidenceByClaim = useMemo(() => {
    const map: Record<string, EvidenceLink[]> = {};
    (answerResponse?.evidence_links ?? []).forEach((link) => {
      map[link.claim_id] = map[link.claim_id] ?? [];
      map[link.claim_id].push(link);
    });
    return map;
  }, [answerResponse]);

  const availableDocIds = useMemo(() => {
    const docs = new Set<string>();
    comparisons.forEach((c) => {
      if (c.doc_id_a) docs.add(c.doc_id_a);
      if (c.doc_id_b) docs.add(c.doc_id_b);
    });
    return Array.from(docs);
  }, [comparisons]);

  const filteredComparisons = useMemo(() => {
    return comparisons.filter((c) => {
      if (filterType !== "all" && c.comparison_type !== filterType) return false;
      if (
        filterDocId !== "all" &&
        c.doc_id_a !== filterDocId &&
        c.doc_id_b !== filterDocId
      )
        return false;
      return true;
    });
  }, [comparisons, filterType, filterDocId]);

  const hasFiltersActive = filterType !== "all" || filterDocId !== "all";

  const typeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    comparisons.forEach((c) => {
      counts[c.comparison_type] = (counts[c.comparison_type] ?? 0) + 1;
    });
    return counts;
  }, [comparisons]);

  const header = (badge?: string) => (
    <PageHeader
      title="Cross-Paper Analysis"
      description="How the papers in your answer relate to each other — agreement, genuine contradiction, and conflicts that come apart once you look at the methods."
      badge={badge}
    />
  );

  if (!answerResponse) {
    return (
      <div className="space-y-6">
        {header()}
        <NoAnswerState onGoToAssistant={onGoToAssistant} />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="space-y-6">
        {header("Request failed")}
        <div
          role="alert"
          className="rounded-xl border border-rose-200 bg-rose-50 p-5 text-[13px] text-rose-800"
        >
          <p className="mb-1 font-bold text-rose-900">
            The backend could not complete this request
          </p>
          <p className="break-words leading-relaxed">{answerResponse.answer}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {header(
        comparisons.length > 0
          ? `${comparisons.length} finding${comparisons.length !== 1 ? "s" : ""}`
          : isAbstained
          ? "Abstained"
          : "No comparisons"
      )}

      {comparisons.length > 0 && (
        <p className="-mt-2 text-[12px] leading-relaxed text-ink-500">
          {sourceNote
            ? `These relationships come from a run you reopened. ${sourceNote}`
            : "These relationships were drawn from the result of the most recent question in your current session."}
        </p>
      )}

      {/* Legend: the four relationship types, defined in plain language. */}
      <section
        aria-label="Comparison categories"
        className="grid grid-cols-1 gap-2.5 rounded-xl border border-ink-200 bg-white p-4 shadow-2xs sm:grid-cols-2"
      >
        {(Object.keys(COMPARISON_META) as ComparisonType[]).map((type) => {
          const meta = COMPARISON_META[type];
          const Icon = meta.icon;
          return (
            <div key={type} className="flex items-start gap-2.5">
              <span className={`pt-chip shrink-0 ${meta.badge}`}>
                <Icon className="h-3 w-3" />
              </span>
              <p className="text-[11px] leading-snug text-ink-500">
                <span className="font-semibold text-ink-700">
                  {meta.shortLabel}.{" "}
                </span>
                {meta.meaning}
              </p>
            </div>
          );
        })}
      </section>

      {comparisons.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setFilterType("all")}
            aria-pressed={filterType === "all"}
            className={`pt-chip border-ink-200 bg-white text-ink-600 transition-opacity ${
              filterType === "all" ? "ring-2 ring-brand-200" : "opacity-80 hover:opacity-100"
            }`}
          >
            All ({comparisons.length})
          </button>
          {(Object.keys(COMPARISON_META) as ComparisonType[]).map((type) => {
            const count = typeCounts[type] ?? 0;
            if (count === 0) return null;
            const meta = COMPARISON_META[type];
            const Icon = meta.icon;
            return (
              <button
                key={type}
                type="button"
                onClick={() => setFilterType(filterType === type ? "all" : type)}
                aria-pressed={filterType === type}
                className={`pt-chip transition-opacity ${meta.badge} ${
                  filterType === type
                    ? "ring-2 ring-brand-200"
                    : "opacity-75 hover:opacity-100"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {meta.shortLabel} ({count})
              </button>
            );
          })}
        </div>
      )}

      {comparisons.length > 1 && availableDocIds.length > 1 && (
        <section
          aria-label="Comparison filters"
          className="rounded-xl border border-ink-200 bg-white p-4 shadow-2xs"
        >
          <div className="mb-2 flex items-center gap-2">
            <span className="pt-eyebrow">Paper</span>
            {hasFiltersActive && (
              <button
                onClick={() => {
                  setFilterType("all");
                  setFilterDocId("all");
                }}
                className="ml-auto text-[11px] font-semibold text-brand-600 hover:text-brand-800"
              >
                Clear filters
              </button>
            )}
          </div>
          <select
            value={filterDocId}
            onChange={(e) => setFilterDocId(e.target.value)}
            className="min-w-[200px] rounded-md border border-ink-200 bg-white px-2 py-1 text-[12px] text-ink-700"
            aria-label="Filter by document"
          >
            <option value="all">All papers</option>
            {availableDocIds.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </section>
      )}

      {isAbstained || comparisons.length === 0 ? (
        <NoComparisonsState isAbstained={isAbstained} />
      ) : filteredComparisons.length === 0 ? (
        <div className="py-12 text-center text-[13px] text-ink-500">
          No comparisons match the current filters.
          {hasFiltersActive && (
            <button
              onClick={() => {
                setFilterType("all");
                setFilterDocId("all");
              }}
              className="ml-2 font-semibold text-brand-600 hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3.5">
          {filteredComparisons.map((comparison, i) => (
            <ComparisonCard
              key={`${comparison.claim_id_a}-${comparison.claim_id_b}-${i}`}
              comparison={comparison}
              index={i}
              evidenceByClaim={evidenceByClaim}
            />
          ))}
        </div>
      )}
    </div>
  );
}
