"use client";

import React, { useState, useMemo } from "react";
import {
  FileSearch,
  Filter,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  ShieldX,
  ShieldAlert,
  HelpCircle,
  Quote,
  Sparkles,
  Info,
} from "lucide-react";
import {
  AnswerResponse,
  EvidenceLink,
  CitationCheck,
  ClaimVerificationStatus,
  EvidenceRelation,
} from "@/lib/types";
import { PageHeader } from "@/components/ui/PageHeader";
import { PaperRef } from "@/components/ui/PaperRef";
import { ResearchHistoryPanel } from "@/components/history/ResearchHistoryPanel";
import {
  ResearchHistoryEntry,
  formatHistoryTime,
} from "@/lib/research-history";

const RELATION_LABELS: Record<EvidenceRelation, string> = {
  supports: "Supports",
  weakens: "Weakens",
  contradicts: "Contradicts",
  contextualizes: "Context",
  insufficient: "Insufficient",
};

const RELATION_COLORS: Record<EvidenceRelation, string> = {
  supports: "border-emerald-200 bg-emerald-50 text-emerald-800",
  weakens: "border-amber-200 bg-amber-50 text-amber-800",
  contradicts: "border-red-200 bg-red-50 text-red-800",
  contextualizes: "border-blue-200 bg-blue-50 text-blue-800",
  insufficient: "border-ink-200 bg-ink-100 text-ink-600",
};

const STATUS_CONFIG: Record<
  ClaimVerificationStatus,
  { label: string; cls: string; icon: React.ReactNode; meaning: string }
> = {
  verified: {
    label: "Verified",
    cls: "border-emerald-200 bg-emerald-50 text-emerald-800",
    icon: <ShieldCheck className="h-3.5 w-3.5" />,
    meaning: "The passage exists in the source and its exact text supports the claim.",
  },
  unsupported: {
    label: "Unsupported",
    cls: "border-amber-200 bg-amber-50 text-amber-800",
    icon: <ShieldX className="h-3.5 w-3.5" />,
    meaning: "The passage was found, but it does not actually establish the claim.",
  },
  invalid_reference: {
    label: "Invalid Reference",
    cls: "border-rose-200 bg-rose-50 text-rose-900",
    icon: <ShieldAlert className="h-3.5 w-3.5" />,
    meaning: "The cited passage is not part of the evidence that was retrieved.",
  },
  unverified: {
    label: "Unverified",
    cls: "border-ink-200 bg-ink-100 text-ink-600",
    icon: <HelpCircle className="h-3.5 w-3.5" />,
    meaning: "Verification did not complete, so this is treated as unconfirmed.",
  },
};

function VerificationBadge({ status }: { status: ClaimVerificationStatus }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.unverified;
  return (
    <span className={`pt-chip ${cfg.cls}`} title={cfg.meaning}>
      {cfg.icon}
      {cfg.label}
    </span>
  );
}

function LayerResult({
  label,
  passed,
  passText,
  failText,
}: {
  label: string;
  passed: boolean | null | undefined;
  passText: string;
  failText: string;
}) {
  if (passed === null || passed === undefined) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded border border-ink-200 bg-ink-100 px-2 py-0.5 text-[11px] text-ink-500">
        <HelpCircle className="h-3 w-3 text-ink-300" />
        <span className="font-semibold text-ink-600">{label}</span>
        <span>not evaluated</span>
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-[11px] font-medium ${
        passed
          ? "border-emerald-200 bg-emerald-50 text-emerald-800"
          : "border-rose-200 bg-rose-50 text-rose-800"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          passed ? "bg-emerald-500" : "bg-rose-500"
        }`}
        aria-hidden="true"
      />
      <span className="font-semibold">{label}</span>
      <span>{passed ? passText : failText}</span>
    </span>
  );
}

function CitationCheckRow({ check }: { check: CitationCheck }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="overflow-hidden rounded-lg border border-ink-200 text-[12px]">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between gap-2 bg-ink-100/60 px-3 py-2 text-left transition-colors hover:bg-ink-100"
        aria-expanded={expanded}
      >
        <span className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
          <span
            className="max-w-[200px] truncate font-mono text-[11px] text-ink-600"
            title={check.chunk_id}
          >
            {check.chunk_id}
          </span>
          <VerificationBadge status={check.status} />
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5 shrink-0 text-ink-400" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-ink-400" />
        )}
      </button>
      {expanded && (
        <div className="space-y-2.5 bg-white px-3 py-3">
          <div className="flex flex-wrap gap-2">
            <LayerResult
              label="Passage retrieved"
              passed={check.exists}
              passText="yes"
              failText="no"
            />
            <LayerResult
              label="Supports claim"
              passed={check.supports}
              passText="yes"
              failText="no"
            />
          </div>
          {check.notes && (
            <p className="rounded border border-ink-100 bg-ink-100/60 px-2 py-1.5 font-mono text-[11px] leading-relaxed text-ink-600">
              {check.notes}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function EvidenceLinkCard({
  link,
  citationChecks,
}: {
  link: EvidenceLink;
  citationChecks: CitationCheck[];
}) {
  const relCls = RELATION_COLORS[link.relation] ?? RELATION_COLORS.insufficient;
  const relLabel = RELATION_LABELS[link.relation] ?? link.relation;
  const check = citationChecks.find((c) => c.chunk_id === link.chunk_id);

  return (
    <div className="overflow-hidden rounded-lg border border-ink-200 bg-white">
      <div className="flex flex-wrap items-center gap-2 border-b border-ink-100 bg-ink-100/50 px-3 py-2">
        <span className={`pt-chip ${relCls}`}>{relLabel}</span>
        {check && <VerificationBadge status={check.status} />}
        {typeof link.confidence === "number" && (
          <span
            className="ml-auto font-mono text-[11px] text-ink-500"
            title="Model-reported confidence that this passage supports the claim"
          >
            {(link.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <div className="space-y-2 px-3 py-3">
        <PaperRef
          docId={link.doc_id}
          page={link.page}
          chunkId={link.chunk_id}
        />

        {link.quote ? (
          <blockquote className="flex items-start gap-2 rounded-md border border-brand-100 bg-brand-50/60 px-3 py-2.5">
            <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-400" />
            <p className="text-[12px] italic leading-relaxed text-ink-900">
              &ldquo;{link.quote}&rdquo;
            </p>
          </blockquote>
        ) : (
          <p className="rounded-md border border-dashed border-ink-200 px-3 py-2 text-[11px] text-ink-500">
            No verbatim passage was returned for this reference.
          </p>
        )}

        {link.rationale && (
          <p className="text-[12px] leading-relaxed text-ink-600">
            {link.rationale}
          </p>
        )}

        {link.conflict_reason && (
          <div className="flex items-start gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-2 text-[11px] text-amber-800">
            <Info className="mt-0.5 h-3 w-3 shrink-0 text-amber-600" />
            <span>
              <span className="font-semibold">Why this looks like a conflict: </span>
              {link.conflict_reason}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function ClaimLedgerCard({
  claimId,
  evidenceLinks,
  citationChecks,
}: {
  claimId: string;
  evidenceLinks: EvidenceLink[];
  citationChecks: CitationCheck[];
}) {
  const [open, setOpen] = useState(true);

  const claimChecks = citationChecks.filter((c) => c.claim_id === claimId);

  const overallStatus: ClaimVerificationStatus = useMemo(() => {
    if (claimChecks.length === 0) return "unverified";
    if (claimChecks.some((c) => c.status === "invalid_reference"))
      return "invalid_reference";
    if (claimChecks.some((c) => c.status === "unsupported")) return "unsupported";
    if (claimChecks.every((c) => c.status === "verified")) return "verified";
    return "unverified";
  }, [claimChecks]);

  return (
    <article
      aria-label={`Claim ${claimId}`}
      className="overflow-hidden rounded-xl border border-ink-200 bg-white shadow-card"
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-b border-ink-100 bg-ink-100/60 px-4 py-3 text-left transition-colors hover:bg-ink-100"
        aria-expanded={open}
      >
        <span className="flex min-w-0 flex-1 flex-wrap items-center gap-2.5">
          <span className="rounded border border-brand-200 bg-brand-50 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-brand-700">
            {claimId}
          </span>
          <VerificationBadge status={overallStatus} />
          <span className="text-[11px] text-ink-500">
            {evidenceLinks.length} source passage
            {evidenceLinks.length !== 1 ? "s" : ""}
          </span>
        </span>
        {open ? (
          <ChevronUp className="h-4 w-4 shrink-0 text-ink-400" />
        ) : (
          <ChevronDown className="h-4 w-4 shrink-0 text-ink-400" />
        )}
      </button>

      {open && (
        <div className="divide-y divide-ink-100">
          <div className="space-y-3 px-4 py-3.5">
            <h4 className="pt-eyebrow">Supporting evidence</h4>
            {evidenceLinks.length > 0 ? (
              evidenceLinks.map((link) => (
                <EvidenceLinkCard
                  key={link.chunk_id}
                  link={link}
                  citationChecks={citationChecks}
                />
              ))
            ) : (
              <p className="text-[12px] italic text-ink-500">
                No evidence link is attached to this claim.
              </p>
            )}
          </div>

          {claimChecks.length > 0 && (
            <div className="space-y-2 px-4 py-3.5">
              <h4 className="pt-eyebrow">
                Verification checks ({claimChecks.length})
              </h4>
              {claimChecks.map((check) => (
                <CitationCheckRow key={check.chunk_id} check={check} />
              ))}
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
        <FileSearch className="h-6 w-6" />
      </div>
      <div className="space-y-1">
        <h2 className="text-[15px] font-semibold text-ink-800">
          No evidence to show yet
        </h2>
        <p className="mx-auto max-w-sm text-[13px] leading-relaxed text-ink-500">
          Ask a research question and every claim it produces will appear here with
          its exact source passage and verification result.
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

function AbstentionLedgerState({ reason }: { reason?: string | null }) {
  return (
    <div
      role="alert"
      className="space-y-3 rounded-xl border border-amber-200 bg-amber-50 p-5 sm:p-6"
    >
      <div className="flex items-center gap-2">
        <ShieldAlert className="h-5 w-5 shrink-0 text-amber-600" />
        <h3 className="text-sm font-bold text-amber-950">
          No verified evidence to display
        </h3>
      </div>
      {reason && (
        <p className="break-words rounded border border-amber-200 bg-white/80 p-3 font-mono text-[12px] leading-relaxed text-amber-900">
          {reason}
        </p>
      )}
      <p className="text-[12px] leading-relaxed text-amber-800">
        The system withheld the answer, so the ledger is intentionally empty. Nothing
        here is presented as verified unless its source passage was confirmed.
      </p>
    </div>
  );
}

export interface EvidenceLedgerViewProps {
  answerResponse: AnswerResponse | null;
  onGoToAssistant?: () => void;
  /** Runs saved in this browser, newest first. */
  historyEntries?: ResearchHistoryEntry[];
  /** id of the saved run currently on screen, when one was reopened. */
  activeHistoryId?: string | null;
  /** True when the on-screen result was reopened from saved research rather than just asked. */
  openedFromHistory?: boolean;
  onOpenHistoryEntry?: (entry: ResearchHistoryEntry) => void;
  onDeleteHistoryEntry?: (id: string) => void;
  onClearHistory?: () => void;
}

export function EvidenceLedgerView({
  answerResponse,
  onGoToAssistant,
  historyEntries = [],
  activeHistoryId = null,
  openedFromHistory = false,
  onOpenHistoryEntry,
  onDeleteHistoryEntry,
  onClearHistory,
}: EvidenceLedgerViewProps) {
  const [filterDocId, setFilterDocId] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterRelation, setFilterRelation] = useState<string>("all");

  const allClaimIds = useMemo(() => {
    if (!answerResponse?.evidence_links) return [];
    const seen = new Set<string>();
    answerResponse.evidence_links.forEach((l) => seen.add(l.claim_id));
    return Array.from(seen);
  }, [answerResponse]);

  const availableDocIds = useMemo(() => {
    if (!answerResponse?.evidence_links) return [];
    const docs = new Set<string>();
    answerResponse.evidence_links.forEach((l) => {
      if (l.doc_id) docs.add(l.doc_id);
    });
    return Array.from(docs);
  }, [answerResponse]);

  const availableStatuses = useMemo(() => {
    if (!answerResponse?.citation_checks) return [];
    const statuses = new Set<string>();
    answerResponse.citation_checks.forEach((c) => statuses.add(c.status));
    return Array.from(statuses);
  }, [answerResponse]);

  const availableRelations = useMemo(() => {
    if (!answerResponse?.evidence_links) return [];
    const rels = new Set<string>();
    answerResponse.evidence_links.forEach((l) => rels.add(l.relation));
    return Array.from(rels);
  }, [answerResponse]);

  const filteredClaimIds = useMemo(() => {
    if (!answerResponse) return [];
    return allClaimIds.filter((claimId) => {
      const links = answerResponse.evidence_links.filter(
        (l) => l.claim_id === claimId
      );
      const checks = answerResponse.citation_checks.filter(
        (c) => c.claim_id === claimId
      );

      if (filterDocId !== "all" && !links.some((l) => l.doc_id === filterDocId))
        return false;
      if (filterStatus !== "all" && !checks.some((c) => c.status === filterStatus))
        return false;
      if (
        filterRelation !== "all" &&
        !links.some((l) => l.relation === filterRelation)
      )
        return false;

      return true;
    });
  }, [allClaimIds, answerResponse, filterDocId, filterStatus, filterRelation]);

  const hasFiltersActive =
    filterDocId !== "all" || filterStatus !== "all" || filterRelation !== "all";

  const clearFilters = () => {
    setFilterDocId("all");
    setFilterStatus("all");
    setFilterRelation("all");
  };

  /**
   * A saved run is on screen either because the workspace says it was reopened, or
   * because it is not the run the assistant just produced (older than the newest).
   */
  const isActiveReplay =
    activeHistoryId !== null &&
    (openedFromHistory || historyEntries[0]?.id !== activeHistoryId);

  const activeEntry = isActiveReplay
    ? historyEntries.find((e) => e.id === activeHistoryId) ?? null
    : null;

  const replayNote = (
    <p className="-mt-2 text-[12px] leading-relaxed text-ink-500">
      {activeEntry
        ? `Reopened from your saved research on ${formatHistoryTime(
            activeEntry.completedAt
          )}. Claims are listed in the order the answer produced them.`
        : "This is the result of the most recent question in your current session. Claims are listed in the order the answer produced them."}
    </p>
  );

  /** The abstention view has no claims, so only the provenance line is relevant. */
  const abstentionProvenanceNote = activeEntry ? (
    <p className="text-[12px] leading-relaxed text-ink-500">
      {`Reopened from your saved research on ${formatHistoryTime(
        activeEntry.completedAt
      )}.`}
    </p>
  ) : null;

  const historyPanel = (
    <ResearchHistoryPanel
      entries={historyEntries}
      activeId={activeHistoryId}
      onOpen={onOpenHistoryEntry}
      onDelete={onDeleteHistoryEntry}
      onClear={onClearHistory}
    />
  );

  const header = (badge?: string) => (
    <PageHeader
      title="Evidence Explorer"
      description="Every claim from the current result, matched to the exact passage it came from, with the verification outcome shown honestly."
      badge={badge}
    />
  );

  if (!answerResponse) {
    return (
      <div className="space-y-6">
        {header()}
        <NoAnswerState onGoToAssistant={onGoToAssistant} />
        {historyPanel}
      </div>
    );
  }

  if (answerResponse.status === "abstained") {
    return (
      <div className="space-y-6">
        {header("Abstained")}
        <div className="space-y-3">
          {abstentionProvenanceNote}
          <AbstentionLedgerState reason={answerResponse.abstain_reason} />
        </div>
        {historyPanel}
      </div>
    );
  }

  if (answerResponse.status === "error") {
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
        {historyPanel}
      </div>
    );
  }

  const verifiedCount = answerResponse.citation_checks.filter(
    (c) => c.status === "verified"
  ).length;
  const totalCount = answerResponse.citation_checks.length;

  return (
    <div className="space-y-5">
      {header(
        `${allClaimIds.length} claim${allClaimIds.length !== 1 ? "s" : ""} · ${verifiedCount}/${totalCount} citations verified`
      )}

      {replayNote}

      {/* Verification legend — statuses are never implied by colour alone. */}
      <section
        aria-label="Verification status legend"
        className="grid grid-cols-1 gap-2 rounded-xl border border-ink-200 bg-white p-3.5 shadow-2xs sm:grid-cols-2 lg:grid-cols-4"
      >
        {(
          Object.keys(STATUS_CONFIG) as ClaimVerificationStatus[]
        ).map((status) => {
          const cfg = STATUS_CONFIG[status];
          return (
            <div key={status} className="flex items-start gap-2">
              <span className={`pt-chip shrink-0 ${cfg.cls}`}>{cfg.icon}</span>
              <p className="text-[11px] leading-snug text-ink-500">{cfg.meaning}</p>
            </div>
          );
        })}
      </section>

      {(availableDocIds.length > 1 ||
        availableStatuses.length > 1 ||
        availableRelations.length > 1) && (
        <section
          aria-label="Evidence filters"
          className="rounded-xl border border-ink-200 bg-white p-4 shadow-2xs"
        >
          <div className="mb-3 flex items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-ink-400" />
            <span className="pt-eyebrow">Filter</span>
            {hasFiltersActive && (
              <button
                onClick={clearFilters}
                className="ml-auto text-[11px] font-semibold text-brand-600 hover:text-brand-800"
              >
                Clear filters
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-3">
            {availableDocIds.length > 1 && (
              <label className="flex min-w-[160px] flex-col gap-1 text-[11px] font-semibold text-ink-500">
                Paper
                <select
                  value={filterDocId}
                  onChange={(e) => setFilterDocId(e.target.value)}
                  className="rounded-md border border-ink-200 bg-white px-2 py-1 text-[12px] font-normal text-ink-700"
                  aria-label="Filter by document"
                >
                  <option value="all">All papers</option>
                  {availableDocIds.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {availableStatuses.length > 1 && (
              <label className="flex min-w-[160px] flex-col gap-1 text-[11px] font-semibold text-ink-500">
                Verification
                <select
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className="rounded-md border border-ink-200 bg-white px-2 py-1 text-[12px] font-normal text-ink-700"
                  aria-label="Filter by verification status"
                >
                  <option value="all">All statuses</option>
                  {availableStatuses.map((s) => (
                    <option key={s} value={s}>
                      {STATUS_CONFIG[s as ClaimVerificationStatus]?.label ?? s}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {availableRelations.length > 1 && (
              <label className="flex min-w-[160px] flex-col gap-1 text-[11px] font-semibold text-ink-500">
                Relation
                <select
                  value={filterRelation}
                  onChange={(e) => setFilterRelation(e.target.value)}
                  className="rounded-md border border-ink-200 bg-white px-2 py-1 text-[12px] font-normal text-ink-700"
                  aria-label="Filter by evidence relation"
                >
                  <option value="all">All relations</option>
                  {availableRelations.map((r) => (
                    <option key={r} value={r}>
                      {RELATION_LABELS[r as EvidenceRelation] ?? r}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
        </section>
      )}

      {filteredClaimIds.length === 0 ? (
        <div className="py-12 text-center text-[13px] text-ink-500">
          No claims match the current filters.
          {hasFiltersActive && (
            <button
              onClick={clearFilters}
              className="ml-2 font-semibold text-brand-600 hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3.5">
          {filteredClaimIds.map((claimId) => (
            <ClaimLedgerCard
              key={claimId}
              claimId={claimId}
              evidenceLinks={answerResponse.evidence_links.filter(
                (l) => l.claim_id === claimId
              )}
              citationChecks={answerResponse.citation_checks}
            />
          ))}
        </div>
      )}

      {historyPanel}
    </div>
  );
}
