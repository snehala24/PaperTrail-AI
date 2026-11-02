"use client";

import React from "react";
import { RefreshCw, AlertTriangle, CheckCircle2 } from "lucide-react";

export interface StatusIndicatorProps {
  status: "connected" | "disconnected" | "checking";
  version?: string;
  onRefresh?: () => void;
  className?: string;
}

export function StatusIndicator({
  status,
  version = "0.1.0",
  onRefresh,
  className = "",
}: StatusIndicatorProps) {
  return (
    <div className={`flex items-center gap-1.5 ${className}`}>
      <div
        className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors duration-150 ${
          status === "connected"
            ? "border-emerald-200 bg-emerald-50 text-emerald-800"
            : status === "disconnected"
            ? "border-rose-200 bg-rose-50 text-rose-800"
            : "border-ink-200 bg-ink-100 text-ink-600"
        }`}
        role="status"
        aria-live="polite"
      >
        {status === "checking" && (
          <span className="flex items-center">
            <RefreshCw className="mr-1.5 h-3 w-3 animate-spin text-ink-400" />
            Connecting...
          </span>
        )}
        {status === "connected" && (
          <span className="flex items-center">
            <CheckCircle2 className="mr-1.5 h-3 w-3 text-emerald-600" />
            Backend v{version} Online
          </span>
        )}
        {status === "disconnected" && (
          <span className="flex items-center">
            <AlertTriangle className="mr-1.5 h-3 w-3 shrink-0 text-rose-500" />
            Backend Offline (:8000)
          </span>
        )}
      </div>

      {onRefresh && (
        <button
          onClick={onRefresh}
          className="rounded-md p-1.5 text-ink-400 transition-colors hover:bg-ink-100 hover:text-ink-700"
          title="Refresh backend status"
          aria-label="Refresh backend connection"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
