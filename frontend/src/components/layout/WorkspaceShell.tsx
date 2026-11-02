"use client";

import React, { useState, useEffect } from "react";
import { AppSidebar, WorkspaceSection, NAV_ITEMS } from "./AppSidebar";
import { AppHeader } from "./AppHeader";
import { HealthResponse } from "@/lib/types";
import { getHealth } from "@/lib/api";
import { useCorpusStats } from "@/lib/corpus";
import { DocumentCatalogueProvider } from "@/lib/document-catalogue";
import { BackendStatusProvider } from "@/lib/backend-status";

export interface WorkspaceShellProps {
  activeSection: WorkspaceSection;
  onSelectSection: (section: WorkspaceSection) => void;
  children: React.ReactNode;
}

export function WorkspaceShell({
  activeSection,
  onSelectSection,
  children,
}: WorkspaceShellProps) {
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");
  const {
    stats: corpus,
    error: corpusError,
    reload: reloadCorpus,
  } = useCorpusStats();

  const checkHealth = async () => {
    setHealthStatus("checking");
    try {
      const res = await getHealth();
      setHealth(res);
      setHealthStatus("connected");
    } catch {
      setHealth(null);
      setHealthStatus("disconnected");
    }
  };

  const handleRefreshHealth = () => {
    checkHealth();
    reloadCorpus();
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // Close the mobile drawer on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isMobileOpen) {
        setIsMobileOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isMobileOpen]);

  const activeItem =
    NAV_ITEMS.find((item) => item.id === activeSection) || NAV_ITEMS[0];

  return (
    <div className="flex h-screen overflow-hidden bg-canvas text-ink-900">
      <AppSidebar
        activeSection={activeSection}
        onSelectSection={onSelectSection}
        isMobileOpen={isMobileOpen}
        onCloseMobile={() => setIsMobileOpen(false)}
        corpus={corpus}
        corpusError={corpusError}
      />

      <div className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <AppHeader
          sectionTitle={activeItem.label}
          sectionSubtitle={activeItem.shortDesc}
          backendStatus={healthStatus}
          backendVersion={health?.version}
          onRefreshHealth={handleRefreshHealth}
          onOpenMobileMenu={() => setIsMobileOpen(true)}
        />

        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <BackendStatusProvider status={healthStatus}>
            <DocumentCatalogueProvider documents={corpus?.documents ?? []}>
              {children}
            </DocumentCatalogueProvider>
          </BackendStatusProvider>
        </main>
      </div>
    </div>
  );
}
