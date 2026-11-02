"use client";

import React, { useState } from "react";
import { WorkspaceShell } from "@/components/layout/WorkspaceShell";
import { WorkspaceSection } from "@/components/layout/AppSidebar";
import { DashboardView, SessionResult } from "@/components/dashboard/DashboardView";
import { ResearchLibraryView } from "@/components/library/ResearchLibraryView";
import { ResearchAssistantView } from "@/components/assistant/ResearchAssistantView";
import { EvidenceLedgerView } from "@/components/evidence/EvidenceLedgerView";
import { CrossPaperComparisonsView } from "@/components/comparisons/CrossPaperComparisonsView";
import { AnswerResponse } from "@/lib/types";
import {
  ResearchHistoryEntry,
  formatHistoryTime,
  historyToAnswerResponse,
  useResearchHistory,
} from "@/lib/research-history";

export default function WorkspaceHome() {
  const [activeSection, setActiveSection] = useState<WorkspaceSection>("dashboard");
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  /** Shared answer response — lifted so Evidence Explorer and Cross-Paper Analysis can read it. */
  const [currentAnswer, setCurrentAnswer] = useState<AnswerResponse | null>(null);
  /** Summary of the last question asked in this browser session, for the Dashboard. */
  const [lastResult, setLastResult] = useState<SessionResult | null>(null);
  /** Which saved run is on screen, when one was reopened from this browser's history. */
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  /** False while the on-screen result is the one the assistant just produced. */
  const [openedFromHistory, setOpenedFromHistory] = useState(false);

  const history = useResearchHistory();

  const navigateTo = (section: WorkspaceSection) => setActiveSection(section);

  const handleAnswerGenerated = (
    response: AnswerResponse | null,
    question?: string
  ) => {
    setCurrentAnswer(response);
    setOpenedFromHistory(false);
    if (!response) {
      setActiveHistoryId(null);
      return;
    }

    const claimCount = new Set(response.evidence_links.map((l) => l.claim_id)).size;
    const completedAt = new Date().toISOString();

    const saved = history.record({
      question: question ?? "",
      response,
      docIds: selectedDocIds,
      completedAt,
    });
    setActiveHistoryId(saved.id);

    setLastResult({
      question: question ?? "",
      status: response.status,
      claimCount,
      comparisonCount: response.comparisons.length,
      completedAt,
    });
  };

  const openSavedRun = (entry: ResearchHistoryEntry) => {
    setCurrentAnswer(historyToAnswerResponse(entry));
    setActiveHistoryId(entry.id);
    setOpenedFromHistory(true);
    setLastResult({
      question: entry.question,
      status: entry.status,
      claimCount: new Set(entry.evidenceLinks.map((l) => l.claim_id)).size,
      comparisonCount: entry.comparisons.length,
      completedAt: entry.completedAt,
    });
    setActiveSection("evidence");
  };

  const reopenedEntry = openedFromHistory
    ? history.entries.find((entry) => entry.id === activeHistoryId) ?? null
    : null;

  return (
    <WorkspaceShell
      activeSection={activeSection}
      onSelectSection={setActiveSection}
    >
      {activeSection === "dashboard" && (
        <section aria-label="Dashboard Section">
          <DashboardView
            onSelectSection={navigateTo}
            lastResult={lastResult}
            onOpenHistoryEntry={openSavedRun}
          />
        </section>
      )}

      {activeSection === "library" && (
        <section aria-label="Research Paper Library Section">
          <ResearchLibraryView
            selectedDocIds={selectedDocIds}
            onSelectedDocIdsChange={setSelectedDocIds}
          />
        </section>
      )}

      {activeSection === "assistant" && (
        <section aria-label="Grounded Research Assistant Section">
          <ResearchAssistantView
            selectedDocIds={selectedDocIds}
            onSelectedDocIdsChange={setSelectedDocIds}
            onAnswerGenerated={handleAnswerGenerated}
          />
        </section>
      )}

      {activeSection === "evidence" && (
        <section aria-labelledby="section-evidence">
          <EvidenceLedgerView
            answerResponse={currentAnswer}
            onGoToAssistant={() => navigateTo("assistant")}
            historyEntries={history.entries}
            activeHistoryId={activeHistoryId}
            openedFromHistory={openedFromHistory}
            onOpenHistoryEntry={openSavedRun}
            onDeleteHistoryEntry={history.remove}
            onClearHistory={history.clear}
          />
        </section>
      )}

      {activeSection === "comparisons" && (
        <section aria-labelledby="section-comparisons">
          <CrossPaperComparisonsView
            answerResponse={currentAnswer}
            onGoToAssistant={() => navigateTo("assistant")}
            sourceNote={
              reopenedEntry
                ? `Reopened from your saved research on ${formatHistoryTime(
                    reopenedEntry.completedAt
                  )}.`
                : null
            }
          />
        </section>
      )}
    </WorkspaceShell>
  );
}
