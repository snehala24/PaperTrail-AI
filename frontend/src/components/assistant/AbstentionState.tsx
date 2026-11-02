"use client";

import React from "react";
import { ShieldAlert, Info, ListChecks } from "lucide-react";

export interface AbstentionStateProps {
  reason?: string | null;
  className?: string;
}

export function AbstentionState({
  reason,
  className = "",
}: AbstentionStateProps) {
  return (
    <div
      role="region"
      aria-label="Abstention notice"
      className={`space-y-4 rounded-xl border border-amber-200 bg-amber-50/70 p-5 shadow-2xs sm:p-6 ${className}`}
    >
      <div className="flex items-start gap-3.5">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-100 text-amber-700"
          aria-hidden="true"
        >
          <ShieldAlert className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <span className="pt-chip border-amber-200 bg-amber-100/80 text-amber-800">
            Grounding safeguard
          </span>
          <h3 className="mt-2 text-[17px] font-bold tracking-tight text-amber-950">
            System Abstained: Insufficient or Unverified Evidence
          </h3>
          <p className="mt-1 max-w-xl text-[13px] leading-relaxed text-amber-800">
            PaperTrail AI declined to produce a factual answer because the retrieved
            passages either did not cover the question or did not pass citation
            verification. An honest gap is reported instead of a plausible guess.
          </p>
        </div>
      </div>

      <div className="space-y-1.5 rounded-lg border border-amber-200/80 bg-white/90 p-4 shadow-2xs">
        <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.08em] text-amber-800">
          <ListChecks className="h-3.5 w-3.5 text-amber-600" />
          <span>What the system reported</span>
        </div>
        <p className="break-words font-mono text-[12px] leading-relaxed text-amber-950">
          {reason ||
            "Evidence in the current corpus does not establish verifiable factual support for this query."}
        </p>
      </div>

      <div className="flex items-start gap-2 pt-0.5 text-[12px] font-medium text-amber-700">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>
          Fail-closed guarantee: no synthetic claims or ungrounded citations were
          generated for this result.
        </span>
      </div>
    </div>
  );
}
