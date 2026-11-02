"use client";

import React, { useState } from "react";
import { Clock, ChevronDown, ChevronUp } from "lucide-react";

export interface AnswerDiagnosticsProps {
  timings?: Record<string, number>;
  className?: string;
}

export function AnswerDiagnostics({
  timings,
  className = "",
}: AnswerDiagnosticsProps) {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  if (!timings || Object.keys(timings).length === 0) return null;

  const totalTime =
    timings.total ?? Object.values(timings).reduce((a, b) => a + b, 0);

  return (
    <div
      className={`overflow-hidden rounded-lg border border-ink-200 bg-white shadow-2xs ${className}`}
    >
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left transition-colors hover:bg-ink-100/60"
        aria-expanded={isOpen}
        aria-label={
          isOpen ? "Collapse response timing" : "Expand response timing"
        }
      >
        <span className="flex items-center gap-2">
          <Clock className="h-3.5 w-3.5 text-ink-400" />
          <span className="text-[12px] font-semibold tracking-tight text-ink-700">
            Response timing
          </span>
          <span className="rounded border border-ink-200 bg-ink-100 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-ink-700">
            {totalTime.toFixed(2)}s
          </span>
        </span>

        <span className="flex items-center gap-1 text-[11px] text-ink-400">
          <span>{isOpen ? "Hide" : "Details"}</span>
          {isOpen ? (
            <ChevronUp className="h-3.5 w-3.5" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" />
          )}
        </span>
      </button>

      {isOpen && (
        <div className="space-y-2.5 border-t border-ink-100 bg-ink-100/40 p-4 pt-3">
          <p className="text-[11px] leading-relaxed text-ink-500">
            Wall-clock time spent in each stage of this request. These are performance
            measurements only — they say nothing about how confident the answer is.
          </p>

          <div className="grid grid-cols-2 gap-2 text-[12px] sm:grid-cols-4">
            {Object.entries(timings).map(([stage, sec]) => (
              <div
                key={stage}
                className="rounded-lg border border-ink-200 bg-white p-2.5"
              >
                <span className="block truncate text-[10px] font-semibold uppercase tracking-[0.06em] text-ink-400">
                  {stage}
                </span>
                <span className="font-mono text-[12px] font-bold text-ink-800">
                  {sec.toFixed(3)}s
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
