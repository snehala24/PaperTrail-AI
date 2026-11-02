"use client";

import React, { useState } from "react";
import {
  Copy,
  Check,
  AlertCircle,
  Loader2,
  Sparkles,
  RotateCcw,
  Layers,
} from "lucide-react";
import { AnswerResponse } from "@/lib/types";
import { ClaimCard } from "./ClaimCard";
import { CitationVerification } from "./CitationVerification";
import { ComparisonPanel } from "./ComparisonPanel";
import { AbstentionState } from "./AbstentionState";
import { AnswerDiagnostics } from "./AnswerDiagnostics";

export interface AnswerPanelProps {
  response: AnswerResponse | null;
  isLoading: boolean;
  error: string | null;
  onReset: () => void;
  className?: string;
}

export function AnswerPanel({
  response,
  isLoading,
  error,
  onReset,
  className = "",
}: AnswerPanelProps) {
  const [copiedAnswer, setCopiedAnswer] = useState<boolean>(false);

  const claimIds = React.useMemo(() => {
    if (!response || !response.evidence_links) return [];
    const set = new Set<string>();
    response.evidence_links.forEach((l) => set.add(l.claim_id));
    return Array.from(set);
  }, [response]);

  const handleCopy = async () => {
    if (!response?.answer) return;
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopiedAnswer(true);
      setTimeout(() => setCopiedAnswer(false), 2000);
    } catch {
      setCopiedAnswer(false);
    }
  };

  // 1. Working
  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Generating answer loading state"
        className={`space-y-3 rounded-xl border border-ink-200 bg-white p-8 text-center shadow-card ${className}`}
      >
        <Loader2 className="mx-auto h-8 w-8 animate-spin text-brand-600" />
        <div className="space-y-1">
          <h3 className="text-[15px] font-bold tracking-tight text-ink-900">
            Synthesizing &amp; Verifying Grounded Answer
          </h3>
          <p className="mx-auto max-w-md text-[13px] leading-relaxed text-ink-500">
            Searching the selected papers, reading the most relevant passages, and
            checking every quotation against its source page before the answer is
            shown to you.
          </p>
        </div>
      </div>
    );
  }

  // 2. Request failed before an answer was produced
  if (error) {
    return (
      <div
        role="alert"
        className={`space-y-3 rounded-xl border border-rose-200 bg-rose-50 p-5 text-[13px] text-rose-800 ${className}`}
      >
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
          <div className="flex-1 space-y-1">
            <h3 className="text-sm font-bold text-rose-900">
              Query Execution Failed
            </h3>
            <p className="leading-relaxed break-words">{error}</p>
            <p className="text-[12px] leading-relaxed text-rose-700">
              No answer was produced. Your previous result is still available in the
              Evidence Explorer.
            </p>
          </div>
        </div>
        <div className="flex justify-end pt-1">
          <button
            type="button"
            onClick={onReset}
            className="rounded-md border border-rose-300 bg-white px-3 py-1.5 text-[12px] font-semibold text-rose-700 transition-colors hover:bg-rose-50"
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  // 3. Nothing asked yet
  if (!response) return null;

  // 4. Backend reported an execution error (distinct from abstention)
  if (response.status === "error") {
    return (
      <div
        role="alert"
        className={`space-y-4 rounded-xl border border-rose-200 bg-rose-50 p-5 sm:p-6 ${className}`}
      >
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-6 w-6 shrink-0 text-rose-600" />
          <div className="space-y-1.5">
            <span className="pt-chip border-rose-200 bg-rose-100 text-rose-800">
              Backend Execution Error
            </span>
            <h3 className="mt-1 text-[15px] font-bold tracking-tight text-rose-950">
              Internal Engine Error
            </h3>
            <p className="break-words text-[13px] leading-relaxed text-rose-800">
              {response.answer}
            </p>
            {response.abstain_reason && (
              <p className="mt-2 break-words rounded border border-rose-200 bg-white/80 p-2.5 font-mono text-[12px] text-rose-900">
                {response.abstain_reason}
              </p>
            )}
          </div>
        </div>
        <AnswerDiagnostics timings={response.timings} />
      </div>
    );
  }

  // 5. Honest abstention
  if (response.status === "abstained") {
    return (
      <div className={`space-y-4 ${className}`}>
        <AbstentionState reason={response.abstain_reason} />
        <AnswerDiagnostics timings={response.timings} />
      </div>
    );
  }

  // 6. Verified answer
  return (
    <div className={`animate-rise-in space-y-5 ${className}`}>
      <section
        aria-label="Synthesized Answer"
        className="space-y-3.5 rounded-xl border border-ink-200 bg-white p-5 shadow-card sm:p-6"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-100 pb-3">
          <span className="pt-chip border-brand-200 bg-brand-50 text-brand-700">
            <Sparkles className="h-3 w-3" />
            Evidence-Grounded Answer
          </span>

          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] font-medium text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800"
            title="Copy answer text"
            aria-label="Copy answer text"
          >
            {copiedAnswer ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-600" />
                <span className="text-emerald-700">Copied</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        <p className="whitespace-pre-line text-[15px] leading-relaxed text-ink-900">
          {response.answer}
        </p>
      </section>

      {claimIds.length > 0 && (
        <section aria-label="Claims and supporting evidence" className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="flex items-center gap-1.5 text-[12px] font-bold uppercase tracking-[0.08em] text-ink-700">
              <Layers className="h-4 w-4 text-brand-600" />
              <span>
                Grounded Claims &amp; Verbatim Evidence ({claimIds.length})
              </span>
            </h4>
            <span className="text-[11px] font-medium text-ink-500">
              Each claim is tied to the exact sentence it came from
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {claimIds.map((cid) => {
              const links = response.evidence_links.filter(
                (l) => l.claim_id === cid
              );
              return (
                <ClaimCard
                  key={cid}
                  claimId={cid}
                  evidenceLinks={links}
                  citationChecks={response.citation_checks}
                />
              );
            })}
          </div>
        </section>
      )}

      {response.citation_checks && response.citation_checks.length > 0 && (
        <section aria-label="Citation verification checks">
          <CitationVerification checks={response.citation_checks} />
        </section>
      )}

      {response.comparisons && response.comparisons.length > 0 && (
        <section aria-label="Cross-paper comparisons">
          <ComparisonPanel comparisons={response.comparisons} />
        </section>
      )}

      <section aria-label="Response timing">
        <AnswerDiagnostics timings={response.timings} />
      </section>

      <div className="flex flex-wrap justify-end gap-2 pt-1">
        <button type="button" onClick={onReset} className="pt-btn-secondary">
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Ask another question</span>
        </button>
      </div>
    </div>
  );
}
