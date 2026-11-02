"use client";

import React from "react";
import {
  ShieldCheck,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
} from "lucide-react";
import { CitationCheck, ClaimVerificationStatus } from "@/lib/types";

export interface CitationVerificationProps {
  checks: CitationCheck[];
  className?: string;
}

const STATUS_CONFIG: Record<
  ClaimVerificationStatus,
  { label: string; cls: string; icon: React.ElementType }
> = {
  verified: {
    label: "Verified",
    cls: "border-emerald-200 bg-emerald-50 text-emerald-800",
    icon: CheckCircle2,
  },
  unsupported: {
    label: "Unsupported",
    cls: "border-red-200 bg-red-50 text-red-800",
    icon: XCircle,
  },
  invalid_reference: {
    label: "Invalid Reference",
    cls: "border-rose-200 bg-rose-50 text-rose-900",
    icon: AlertTriangle,
  },
  unverified: {
    label: "Unverified",
    cls: "border-ink-200 bg-ink-100 text-ink-600",
    icon: HelpCircle,
  },
};

function CheckItem({
  passed,
  label,
}: {
  passed: boolean | null;
  label: string;
}) {
  if (passed === null || passed === undefined) {
    return (
      <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-500">
        <HelpCircle className="h-3.5 w-3.5 text-ink-300" />
        <span>{label}</span>
        <span className="font-mono text-[11px] text-ink-400">— not evaluated</span>
      </span>
    );
  }
  const Icon = passed ? CheckCircle2 : XCircle;
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[12px] font-medium ${
        passed ? "text-emerald-700" : "text-red-700"
      }`}
    >
      <Icon className={`h-3.5 w-3.5 ${passed ? "text-emerald-600" : "text-red-600"}`} />
      <span>{label}</span>
    </span>
  );
}

export function CitationVerification({
  checks,
  className = "",
}: CitationVerificationProps) {
  if (!checks || checks.length === 0) return null;

  const verifiedCount = checks.filter((c) => c.status === "verified").length;

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="flex items-center gap-1.5 text-[12px] font-bold uppercase tracking-[0.08em] text-ink-700">
          <ShieldCheck className="h-4 w-4 text-brand-600" />
          <span>3-Layer Citation Verification Audit</span>
        </h4>
        <span className="text-[11px] font-medium text-ink-500">
          {verifiedCount} of {checks.length} citation{checks.length === 1 ? "" : "s"} fully
          verified
        </span>
      </div>

      <div className="space-y-2">
        {checks.map((check, idx) => {
          const statusCfg = STATUS_CONFIG[check.status] ?? STATUS_CONFIG.unverified;
          const StatusIcon = statusCfg.icon;

          return (
            <div
              key={`${check.claim_id}_${check.chunk_id}_${idx}`}
              className="space-y-2 rounded-lg border border-ink-200 bg-white p-3 shadow-2xs"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-100 pb-2">
                <div className="flex min-w-0 items-center gap-2 text-[12px]">
                  <span className="rounded bg-ink-100 px-1.5 py-0.5 font-mono font-semibold text-ink-700">
                    {check.claim_id}
                  </span>
                  <span className="text-ink-300" aria-hidden="true">
                    &rarr;
                  </span>
                  <span className="truncate rounded border border-brand-200 bg-brand-50 px-1.5 py-0.5 font-mono text-[11px] text-brand-700">
                    {check.chunk_id}
                  </span>
                </div>

                <span className={`pt-chip ${statusCfg.cls}`}>
                  <StatusIcon className="h-3.5 w-3.5" />
                  <span>{statusCfg.label}</span>
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <CheckItem
                  passed={check.exists}
                  label={
                    check.exists
                      ? "Retrieved in Context"
                      : "Missing from Evidence Set"
                  }
                />
                <CheckItem
                  passed={check.supports ?? null}
                  label={
                    check.supports === true
                      ? "Entails Claim"
                      : check.supports === false
                      ? "Fails Entailment"
                      : "Claim support"
                  }
                />
              </div>

              {check.notes && (
                <p className="rounded border border-ink-100 bg-ink-100/60 p-2 font-mono text-[11px] leading-relaxed text-ink-600">
                  {check.notes}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
