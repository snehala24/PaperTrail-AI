"use client";

import React from "react";
import { Menu } from "lucide-react";
import { StatusIndicator } from "../ui/StatusIndicator";

export interface AppHeaderProps {
  sectionTitle: string;
  sectionSubtitle: string;
  backendStatus: "connected" | "disconnected" | "checking";
  backendVersion?: string;
  onRefreshHealth: () => void;
  onOpenMobileMenu: () => void;
}

export function AppHeader({
  sectionTitle,
  sectionSubtitle,
  backendStatus,
  backendVersion,
  onRefreshHealth,
  onOpenMobileMenu,
}: AppHeaderProps) {
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between gap-3 border-b border-ink-200 bg-white/85 px-4 py-2.5 backdrop-blur-md sm:px-6">
      <div className="flex min-w-0 items-center gap-2.5">
        <button
          onClick={onOpenMobileMenu}
          className="shrink-0 rounded-lg p-2 text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-800 md:hidden"
          aria-label="Open navigation menu"
        >
          <Menu className="h-5 w-5" />
        </button>

        <nav aria-label="Breadcrumb" className="min-w-0">
          <ol className="flex items-center gap-1.5 text-[11px] text-ink-500">
            <li className="hidden sm:inline">Workspace</li>
            <li className="hidden text-ink-300 sm:inline" aria-hidden="true">
              /
            </li>
            <li className="truncate font-semibold text-ink-800">
              {sectionTitle}
            </li>
            <li
              className="hidden truncate text-ink-400 lg:inline"
              title={sectionSubtitle}
            >
              <span className="mx-1.5 text-ink-300" aria-hidden="true">
                ·
              </span>
              {sectionSubtitle}
            </li>
          </ol>
        </nav>
      </div>

      <div className="shrink-0">
        <StatusIndicator
          status={backendStatus}
          version={backendVersion}
          onRefresh={onRefreshHealth}
        />
      </div>
    </header>
  );
}
