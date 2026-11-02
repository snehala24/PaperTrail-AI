"use client";

import React, { useState } from "react";
import {
  Quote,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  ShieldX,
  ShieldAlert,
  HelpCircle,
} from "lucide-react";
import {
  EvidenceLink,
  CitationCheck,
  EvidenceRelation,
  ClaimVerificationStatus,
} from "@/lib/types";
import { PaperRef } from "@/components/ui/PaperRef";

export interface ClaimCardProps {
  claimId: string;
  claimText?: string;
  evidenceLinks: EvidenceLink[];
  citationChecks?: CitationCheck[];
  className?: string;
}

const RELATION_STYLES: Record<EvidenceRelation, { label: string; badge: string }> = {
  supports: { label: "Supports", badge: "border-emerald-200 bg-emerald-50 text-emerald-800" },
  weakens: { label: "Weakens", badge: "border-amber-200 bg-amber-50 text-amber-800" },
  contradicts: { label: "Contradicts", badge: "border-red-200 bg-red-50 text-red-800" },
  contextualizes: { label: "Context", badge: "border-blue-200 bg-blue-50 text-blue-800" },
  insufficient: { label: "Insufficient", badge: "border-ink-200 bg-ink-100 text-ink-600" },
};

const STATUS_BADGES: Record<
  ClaimVerificationStatus,
  { label: string; cls: string; icon: React.ReactNode }
> = {
  verified: {
    label: "Verified",
    cls: "border-emerald-200 bg-emerald-50 text-emerald-800",
    icon: <ShieldCheck className="h-3 w-3" />,
  },
  unsupported: {
    label: "Unsupported",
    cls: "border-amber-200 bg-amber-50 text-amber-800",
    icon: <ShieldX className="h-3 w-3" />,
  },
  invalid_reference: {
    label: "Invalid Reference",
    cls: "border-rose-200 bg-rose-50 text-rose-900",
    icon: <ShieldAlert className="h-3 w-3" />,
  },
  unverified: {
    label: "Unverified",
    cls: "border-ink-200 bg-ink-100 text-ink-600",
    icon: <HelpCircle className="h-3 w-3" />,
  },
};

/** Worst-case status across a claim's citations — a claim is only as strong as its weakest reference. */
const STATUS_PRIORITY: ClaimVerificationStatus[] = [
  "invalid_reference",
  "unsupported",
  "unverified",
  "verified",
];

export function ClaimCard({
  claimId,
  claimText,
  evidenceLinks,
  citationChecks = [],
  className = "",
}: ClaimCardProps) {
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  const matchingChecks = citationChecks.filter((c) => c.claim_id === claimId);
  const claimStatus: ClaimVerificationStatus =
    matchingChecks.length > 0
      ? STATUS_PRIORITY.find((s) => matchingChecks.some((c) => c.status === s)) ??
        "unverified"
      : "unverified";
  const badge = STATUS_BADGES[claimStatus];

  return (
    <article
      className={`overflow-hidden rounded-xl border border-ink-200 bg-white shadow-card ${className}`}
      aria-label={`Claim ${claimId}`}
    >
      <div className="flex items-start justify-between gap-3 border-b border-ink-100 bg-ink-100/50 p-4">
        <div className="min-w-0 space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded border border-brand-200 bg-brand-50 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-brand-700">
              {claimId}
            </span>
            {matchingChecks.length > 0 && (
              <span
                className={`pt-chip ${badge.cls}`}
                title={
                  claimStatus === "verified"
                    ? "Every cited passage was found in the source and supports this claim."
                    : "At least one citation for this claim did not fully verify."
                }
              >
                {badge.icon}
                {badge.label}
              </span>
            )}
          </div>

          {claimText && (
            <p className="pt-1 text-[14px] font-semibold leading-snug text-ink-900">
              {claimText}
            </p>
          )}
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="shrink-0 rounded-md p-1 text-ink-400 transition-colors hover:bg-ink-100 hover:text-ink-700"
          aria-expanded={isExpanded}
          aria-label={
            isExpanded ? "Collapse claim evidence" : "Expand claim evidence"
          }
        >
          {isExpanded ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </button>
      </div>

      {isExpanded && (
        <div className="space-y-2.5 p-4">
          {evidenceLinks.length === 0 ? (
            <p className="text-[12px] italic text-ink-500">
              No source passage is attached to this claim.
            </p>
          ) : (
            evidenceLinks.map((link, idx) => {
              const relStyle =
                RELATION_STYLES[link.relation] ?? RELATION_STYLES.insufficient;

              return (
                <div
                  key={`${link.chunk_id}_${idx}`}
                  className="space-y-2.5 rounded-lg border border-ink-200 bg-ink-100/40 p-3"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <PaperRef
                      docId={link.doc_id}
                      page={link.page}
                      chunkId={link.chunk_id}
                      className="min-w-0 flex-1"
                    />
                    <div className="flex shrink-0 items-center gap-2">
                      <span className={`pt-chip ${relStyle.badge}`}>
                        {relStyle.label}
                      </span>
                      {typeof link.confidence === "number" && (
                        <span
                          className="font-mono text-[11px] text-ink-500"
                          title="Model-reported confidence that this passage supports the claim"
                        >
                          {(link.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>

                  {link.quote && (
                    <blockquote className="flex items-start gap-2 rounded-md border border-ink-200 bg-white px-3 py-2.5">
                      <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-brand-400" />
                      <p className="text-[12px] italic leading-relaxed text-ink-800">
                        &ldquo;{link.quote}&rdquo;
                      </p>
                    </blockquote>
                  )}

                  {link.rationale && (
                    <p className="text-[12px] leading-relaxed text-ink-600">
                      <span className="font-semibold text-ink-700">
                        Why it matters:{" "}
                      </span>
                      {link.rationale}
                    </p>
                  )}

                  {link.conflict_reason && (
                    <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-2 text-[12px] text-amber-900">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
                      <p>
                        <span className="font-semibold">
                          Why this looks like a conflict:{" "}
                        </span>
                        {link.conflict_reason}
                      </p>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
    </article>
  );
}
