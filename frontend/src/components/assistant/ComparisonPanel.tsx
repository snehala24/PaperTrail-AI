"use client";

import React from "react";
import { GitCompare, AlertTriangle, ArrowRightLeft } from "lucide-react";
import { ClaimComparison } from "@/lib/types";
import { comparisonMeta } from "@/lib/comparison-meta";
import { PaperRef } from "@/components/ui/PaperRef";

export interface ComparisonPanelProps {
  comparisons: ClaimComparison[];
  className?: string;
}

export function ComparisonPanel({
  comparisons,
  className = "",
}: ComparisonPanelProps) {
  if (!comparisons || comparisons.length === 0) return null;

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="flex items-center gap-1.5 text-[12px] font-bold uppercase tracking-[0.08em] text-ink-700">
          <GitCompare className="h-4 w-4 text-brand-600" />
          <span>
            Cross-Paper Comparative Synthesis ({comparisons.length})
          </span>
        </h4>
        <span className="text-[11px] font-medium text-ink-500">
          Agreement, contradiction, and apparent conflict
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {comparisons.map((comp, idx) => {
          const meta = comparisonMeta(comp.comparison_type);
          const Icon = meta.icon;

          return (
            <div
              key={`${comp.claim_id_a}_${comp.claim_id_b}_${idx}`}
              className="space-y-3 rounded-xl border border-ink-200 bg-white p-4 shadow-card"
            >
              <div className="flex flex-wrap items-start justify-between gap-2 border-b border-ink-100 pb-2.5">
                <div className="flex min-w-0 flex-wrap items-center gap-2 text-[12px]">
                  <span className="rounded bg-ink-100 px-1.5 py-0.5 font-mono font-semibold text-ink-700">
                    {comp.claim_id_a}
                  </span>
                  <ArrowRightLeft className="h-3.5 w-3.5 shrink-0 text-ink-300" />
                  <span className="rounded bg-ink-100 px-1.5 py-0.5 font-mono font-semibold text-ink-700">
                    {comp.claim_id_b}
                  </span>
                </div>

                <span className={`pt-chip shrink-0 ${meta.badge}`}>
                  <Icon className="h-3.5 w-3.5" />
                  <span>{meta.label}</span>
                </span>
              </div>

              {/* Both sides of the comparison, named by paper rather than by hash */}
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <div className="rounded-lg border border-ink-200 bg-ink-100/40 px-3 py-2">
                  <p className="pt-eyebrow mb-1">Claim A source</p>
                  <PaperRef docId={comp.doc_id_a} />
                </div>
                <div className="rounded-lg border border-ink-200 bg-ink-100/40 px-3 py-2">
                  <p className="pt-eyebrow mb-1">Claim B source</p>
                  <PaperRef docId={comp.doc_id_b} />
                </div>
              </div>

              {comp.summary && (
                <p className="text-[13px] font-medium leading-relaxed text-ink-800">
                  {comp.summary}
                </p>
              )}

              {comp.conflict_reason && (
                <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5 text-[12px] text-amber-900">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
                  <p>
                    <span className="font-semibold">Why these differ: </span>
                    {comp.conflict_reason}
                  </p>
                </div>
              )}

              <p className="text-[11px] italic leading-relaxed text-ink-500">
                {meta.meaning}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
