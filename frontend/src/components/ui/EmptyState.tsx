import React from "react";
import { LucideIcon } from "lucide-react";

export interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  /** Optional short line shown above the title, e.g. "No papers yet". */
  hint?: string;
  features?: string[];
  featuresHeading?: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  hint,
  features = [],
  featuresHeading = "What you can do",
  actionLabel,
  onAction,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-300 bg-white/70 p-8 text-center shadow-2xs md:p-12 ${className}`}
    >
      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl border border-ink-200 bg-ink-100 text-ink-500">
        <Icon className="h-5 w-5" />
      </div>

      {hint && (
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400">
          {hint}
        </p>
      )}

      <h2 className="max-w-md text-[15px] font-semibold text-ink-900">{title}</h2>

      <p className="mt-1.5 max-w-lg text-[13px] leading-relaxed text-ink-500">
        {description}
      </p>

      {features.length > 0 && (
        <div className="mt-6 w-full max-w-md rounded-lg border border-ink-200 bg-ink-100/60 p-4 text-left">
          <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-500">
            {featuresHeading}
          </h3>
          <ul className="space-y-1.5 text-[13px] text-ink-600">
            {features.map((feat, idx) => (
              <li key={idx} className="flex items-start">
                <span className="mr-2 mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
                <span>{feat}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {actionLabel && onAction && (
        <button onClick={onAction} className="pt-btn-primary mt-6">
          {actionLabel}
        </button>
      )}
    </div>
  );
}
