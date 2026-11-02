"use client";

import React from "react";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  XCircle,
} from "lucide-react";
import { ComparisonType } from "./types";

export interface ComparisonMeta {
  /** Full label used in headers and legends. */
  label: string;
  /** Compact label for chips and filters. */
  shortLabel: string;
  icon: React.ElementType;
  /** Border + background + text classes for badges. */
  badge: string;
  /** Softer surface used behind a card body. */
  surface: string;
  header: string;
  /** Plain-language definition shown in the legend and on each card. */
  meaning: string;
}

/**
 * The four relationships the cross-paper engine can report. The wording is
 * deliberately explicit about the difference between a real contradiction and
 * two papers that merely measure different things — methodological divergence
 * is not evidence of conflict.
 */
export const COMPARISON_META: Record<ComparisonType, ComparisonMeta> = {
  supports: {
    label: "Supports / Harmonious",
    shortLabel: "Supports",
    icon: CheckCircle2,
    badge: "border-emerald-200 bg-emerald-50 text-emerald-800",
    surface: "bg-emerald-50/40",
    header: "bg-emerald-50",
    meaning:
      "Findings from both papers point the same way and reinforce each other.",
  },
  contradicts: {
    label: "Contradiction",
    shortLabel: "Contradicts",
    icon: XCircle,
    badge: "border-red-200 bg-red-50 text-red-800",
    surface: "bg-red-50/40",
    header: "bg-red-50",
    meaning:
      "The papers make incompatible claims under comparable conditions, so one of them must be wrong.",
  },
  apparent_conflict: {
    label: "Apparent Conflict",
    shortLabel: "Apparent Conflict",
    icon: AlertTriangle,
    badge: "border-amber-200 bg-amber-50 text-amber-800",
    surface: "bg-amber-50/40",
    header: "bg-amber-50",
    meaning:
      "The results look opposed, but the difference is explained by different methods, assumptions, datasets or settings — not by a factual disagreement.",
  },
  unresolved: {
    label: "Unresolved / Divergent Context",
    shortLabel: "Unresolved",
    icon: HelpCircle,
    badge: "border-ink-200 bg-ink-100 text-ink-600",
    surface: "bg-ink-100/50",
    header: "bg-ink-100/60",
    meaning:
      "The evidence does not establish how the two claims relate. Left unresolved rather than forced into a verdict.",
  },
};

/** Never crash on an unknown type from the backend — degrade to the neutral one. */
export function comparisonMeta(type: string): ComparisonMeta {
  return COMPARISON_META[type as ComparisonType] ?? COMPARISON_META.unresolved;
}
