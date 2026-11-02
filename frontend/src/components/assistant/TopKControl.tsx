"use client";

import React from "react";
import { Sliders } from "lucide-react";

export interface TopKControlProps {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  className?: string;
}

export function TopKControl({
  value,
  onChange,
  disabled = false,
  className = "",
}: TopKControlProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const num = parseInt(e.target.value, 10);
    if (!isNaN(num)) {
      // Strictly clamped to the range the backend accepts (1–10).
      const clamped = Math.max(1, Math.min(10, num));
      onChange(clamped);
    }
  };

  return (
    <div className={`space-y-1.5 ${className}`}>
      <div className="flex items-center justify-between gap-3">
        <label
          htmlFor="top-k-slider"
          className="flex cursor-pointer items-center gap-1.5 text-[12px] font-semibold text-ink-700"
        >
          <Sliders className="h-3.5 w-3.5 text-brand-600" />
          <span>Evidence depth</span>
        </label>
        <span
          className="rounded border border-ink-200 bg-ink-100 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-ink-700"
          aria-label={`Current top_k value: ${value}`}
        >
          {value} passages
        </span>
      </div>

      <div className="flex items-center gap-3">
        <input
          id="top-k-slider"
          type="range"
          min="1"
          max="10"
          step="1"
          value={value}
          disabled={disabled}
          onChange={handleChange}
          aria-valuemin={1}
          aria-valuemax={10}
          aria-valuenow={value}
          aria-label="Evidence depth (top_k)"
          className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-ink-200 accent-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
        />
        <input
          type="number"
          min="1"
          max="10"
          value={value}
          disabled={disabled}
          onChange={handleChange}
          aria-label="Direct top_k number input"
          className="w-12 rounded-md border border-ink-200 bg-white px-1.5 py-1 text-center font-mono text-[11px] font-semibold text-ink-800 disabled:opacity-50"
        />
      </div>

      <p className="text-[11px] leading-snug text-ink-500">
        How many source passages are considered before answering (1–10, default 8).
        Higher values widen the evidence reviewed; lower values keep it focused.
      </p>
    </div>
  );
}
