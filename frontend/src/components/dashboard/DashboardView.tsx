"use client";

import React from "react";
import {
  ArrowRight,
  BookOpen,
  FileSearch,
  GitCompare,
  Layers,
  FileText,
  ShieldCheck,
  Sparkles,
  Upload,
  AlertCircle,
  Clock,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import {
  WorkspaceSection,
  corpusLabel,
} from "@/components/layout/AppSidebar";
import { useCorpusStats } from "@/lib/corpus";
import { useBackendStatus, BackendStatus } from "@/lib/backend-status";
import { documentDisplayTitle } from "@/lib/display";
import { AnswerStatus } from "@/lib/types";
import { ResearchHistoryEntry, useResearchHistory } from "@/lib/research-history";
import { ResearchHistoryPanel } from "@/components/history/ResearchHistoryPanel";

/** What the assistant produced for the most recent question in this session. */
export interface SessionResult {
  question: string;
  status: AnswerStatus;
  claimCount: number;
  comparisonCount: number;
  completedAt: string;
}

export interface DashboardViewProps {
  onSelectSection: (section: WorkspaceSection) => void;
  /** Result of the last question asked in this browser session, if any. */
  lastResult?: SessionResult | null;
  /** Overrides the workspace connectivity state; defaults to the shell's value. */
  backendStatus?: BackendStatus;
  /** Overrides the locally saved research runs; defaults to this browser's store. */
  historyEntries?: ResearchHistoryEntry[];
  onOpenHistoryEntry?: (entry: ResearchHistoryEntry) => void;
  onDeleteHistoryEntry?: (id: string) => void;
  onClearHistory?: () => void;
  className?: string;
}

const STATUS_PRESENTATION: Record<
  AnswerStatus,
  { label: string; tone: string }
> = {
  answered: { label: "Answered with verified evidence", tone: "text-emerald-700" },
  abstained: { label: "Abstained — evidence not verifiable", tone: "text-amber-700" },
  error: { label: "Request failed", tone: "text-rose-700" },
};

/** Session-local timestamp, shown only for a result produced in this browser session. */
function formatSessionTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return `Completed at ${d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })}`;
}

interface StatCardProps {  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  isLoading: boolean;
}

function StatCard({ icon, label, value, detail, isLoading }: StatCardProps) {
  return (
    <div className="pt-card p-4">
      <div className="flex items-center gap-2 text-ink-500">
        <span className="text-brand-600">{icon}</span>
        <span className="pt-eyebrow">{label}</span>
      </div>
      {isLoading ? (
        <div className="mt-3 space-y-2" aria-hidden="true">
          <div className="h-6 w-16 animate-pulse rounded bg-ink-100" />
          <div className="h-3 w-24 animate-pulse rounded bg-ink-100" />
        </div>
      ) : (
        <>
          <p className="mt-2 text-2xl font-bold tabular-nums tracking-tight text-ink-900">
            {value}
          </p>
          <p className="mt-0.5 text-[12px] leading-relaxed text-ink-500">{detail}</p>
        </>
      )}
    </div>
  );
}

interface QuickActionProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
}

function QuickAction({ icon, title, description, onClick }: QuickActionProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex items-start gap-3 rounded-lg border border-ink-200 bg-white p-3.5 text-left shadow-2xs transition-all duration-150 hover:border-brand-200 hover:bg-brand-50/40 hover:shadow-card-hover"
    >
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-100 text-ink-600 transition-colors group-hover:bg-brand-100 group-hover:text-brand-700">
        {icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-1.5 text-[13px] font-semibold text-ink-900">
          {title}
          <ArrowRight className="h-3.5 w-3.5 shrink-0 text-ink-300 transition-transform duration-150 group-hover:translate-x-0.5 group-hover:text-brand-500" />
        </span>
        <span className="mt-0.5 block text-[12px] leading-relaxed text-ink-500">
          {description}
        </span>
      </span>
    </button>
  );
}

export function DashboardView({
  onSelectSection,
  lastResult = null,
  backendStatus: backendStatusProp,
  historyEntries: historyEntriesProp,
  onOpenHistoryEntry,
  onDeleteHistoryEntry,
  onClearHistory,
  className = "",
}: DashboardViewProps) {
  const contextStatus = useBackendStatus();
  const backendStatus = backendStatusProp ?? contextStatus;
  const { stats, isLoading, error } = useCorpusStats();
  const store = useResearchHistory();
  const historyEntries = historyEntriesProp ?? store.entries;

  const papers = stats?.documents ?? [];
  const paperCount = papers.length;
  const chunkTotal = stats?.totalChunks ?? 0;

  return (
    <div className={`animate-rise-in space-y-6 ${className}`}>
      <PageHeader
        title="Dashboard"
        description="Your research corpus, retrieval coverage, and the evidence status of the work in this session."
        badge={
          backendStatus === "connected"
            ? "Backend connected"
            : backendStatus === "disconnected"
            ? "Backend offline"
            : "Connecting"
        }
      />

      {backendStatus === "disconnected" && (
        <div
          role="alert"
          className="flex items-start gap-2.5 rounded-lg border border-rose-200 bg-rose-50 px-3.5 py-3 text-[12px] text-rose-800"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
          <div>
            <p className="font-semibold">The research backend is not reachable.</p>
            <p className="mt-0.5 leading-relaxed text-rose-700">
              Counts below reflect what the service reported on its last successful
              response. Start the backend on port 8000, then use the refresh action in
              the header.
            </p>
          </div>
        </div>
      )}

      {/* Live corpus metrics — every number comes from the backend. */}
      <section aria-label="Corpus metrics" className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={<BookOpen className="h-4 w-4" />}
          label="Papers"
          value={String(paperCount)}
          detail={
            error
              ? "Unavailable — backend not reached"
              : paperCount === 1
              ? "document in the index"
              : "documents in the index"
          }
          isLoading={isLoading}
        />
        <StatCard
          icon={<FileText className="h-4 w-4" />}
          label="Pages parsed"
          value={String(stats?.totalPages ?? 0)}
          detail={error ? "Unavailable" : "extracted across all papers"}
          isLoading={isLoading}
        />
        <StatCard
          icon={<Layers className="h-4 w-4" />}
          label="Searchable chunks"
          value={error ? "—" : String(chunkTotal)}
          detail={error ? "Unavailable" : "retrieval units, page-anchored"}
          isLoading={isLoading}
        />
        <StatCard
          icon={<ShieldCheck className="h-4 w-4" />}
          label="Answer policy"
          value="Verify or abstain"
          detail="Every claim is cited or the answer is withheld"
          isLoading={false}
        />
      </section>

      {/* Quick actions */}
      <section aria-label="Quick actions" className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <QuickAction
          icon={<Sparkles className="h-4 w-4" />}
          title="Ask the Research Assistant"
          description="Grounded answers with verbatim citations, or an honest abstention."
          onClick={() => onSelectSection("assistant")}
        />
        <QuickAction
          icon={<Upload className="h-4 w-4" />}
          title="Add a paper"
          description="Ingest a PDF and inspect how it is parsed and segmented."
          onClick={() => onSelectSection("library")}
        />
        <QuickAction
          icon={<FileSearch className="h-4 w-4" />}
          title="Explore evidence"
          description="Audit each claim against its source passage and page."
          onClick={() => onSelectSection("evidence")}
        />
        <QuickAction
          icon={<GitCompare className="h-4 w-4" />}
          title="Compare papers"
          description="Separate real contradictions from method differences."
          onClick={() => onSelectSection("comparisons")}
        />
      </section>

      {/* Session activity — only rendered when something actually happened. */}
      <section aria-label="Session activity" className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <div className="pt-card p-5 lg:col-span-2">
          <h2 className="flex items-center gap-2 text-[13px] font-bold text-ink-900">
            <Clock className="h-4 w-4 text-ink-400" />
            This session
          </h2>

          {lastResult ? (
            <div className="mt-3 space-y-3">
              <p className="line-clamp-3 text-[13px] leading-relaxed text-ink-700">
                &ldquo;{lastResult.question}&rdquo;
              </p>
              <p className={`text-[12px] font-semibold ${STATUS_PRESENTATION[lastResult.status]?.tone ?? "text-ink-600"}`}>
                {STATUS_PRESENTATION[lastResult.status]?.label ?? lastResult.status}
              </p>
              <p className="mt-0.5 font-mono text-[11px] text-ink-400">
                {formatSessionTime(lastResult.completedAt)}
              </p>
              <dl className="grid grid-cols-2 gap-2 border-t border-ink-200 pt-3 text-[12px]">
                <div>
                  <dt className="text-ink-500">Claims</dt>
                  <dd className="mt-0.5 font-mono text-[13px] font-semibold text-ink-900">
                    {lastResult.claimCount}
                  </dd>
                </div>
                <div>
                  <dt className="text-ink-500">Cross-paper findings</dt>
                  <dd className="mt-0.5 font-mono text-[13px] font-semibold text-ink-900">
                    {lastResult.comparisonCount}
                  </dd>
                </div>
              </dl>
              <button
                type="button"
                onClick={() => onSelectSection("evidence")}
                className="pt-btn-secondary w-full"
              >
                <FileSearch className="h-3.5 w-3.5" />
                Review the evidence
              </button>
            </div>
          ) : (
            <p className="mt-3 text-[12px] leading-relaxed text-ink-500">
              No questions have been run in this session yet. Anything you do ask is
              saved in this browser, so you can return to it later from the Evidence
              Explorer.
            </p>
          )}
        </div>

        {/* Index table — real stored documents, no invented ordering claims. */}
        <div className="pt-card overflow-hidden lg:col-span-3">
          <div className="flex items-center justify-between gap-3 border-b border-ink-200 px-5 py-3.5">
            <h2 className="text-[13px] font-bold text-ink-900">Indexed papers</h2>
            <button
              type="button"
              onClick={() => onSelectSection("library")}
              className="pt-btn-ghost text-[12px]"
            >
              Open library
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {isLoading ? (
            <ul className="divide-y divide-ink-100" aria-hidden="true">
              {[0, 1, 2].map((i) => (
                <li key={i} className="px-5 py-3">
                  <div className="h-3.5 w-2/3 animate-pulse rounded bg-ink-100" />
                  <div className="mt-2 h-3 w-1/3 animate-pulse rounded bg-ink-100" />
                </li>
              ))}
            </ul>
          ) : error ? (
            <p className="px-5 py-8 text-center text-[12px] text-amber-700">
              The paper list could not be loaded from the backend. This is a
              connectivity issue, not an empty library.
            </p>
          ) : papers.length === 0 ? (
            <div className="px-5 py-10 text-center">
              <p className="text-[13px] font-semibold text-ink-700">
                No papers indexed yet
              </p>
              <p className="mx-auto mt-1 max-w-sm text-[12px] leading-relaxed text-ink-500">
                Add a PDF from the Research Library to make it searchable and
                citable.
              </p>
              <button
                type="button"
                onClick={() => onSelectSection("library")}
                className="pt-btn-primary mt-4"
              >
                <Upload className="h-3.5 w-3.5" />
                Add a paper
              </button>
            </div>
          ) : (
            <ul className="divide-y divide-ink-100">
              {papers.map((doc) => (
                <li
                  key={doc.doc_id}
                  className="flex items-center justify-between gap-3 px-5 py-3 transition-colors hover:bg-ink-100/50"
                >
                  <div className="min-w-0">
                    <p className="truncate text-[13px] font-medium text-ink-900">
                      {documentDisplayTitle(doc)}
                    </p>
                    <p className="mt-0.5 truncate font-mono text-[11px] text-ink-400">
                      {corpusLabel(doc.filename, doc.doc_id)}
                    </p>
                  </div>
                  <div className="shrink-0 text-right font-mono text-[11px] text-ink-500">
                    <span>{doc.page_count} pages</span>
                    <span className="mx-1.5 text-ink-300">•</span>
                    <span>{doc.doc_id}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      {/* Saved runs are only ever real ones from this browser. */}
      {historyEntries.length > 0 && (
        <ResearchHistoryPanel
          entries={historyEntries}
          onOpen={onOpenHistoryEntry}
          onDelete={onDeleteHistoryEntry ?? store.remove}
          onClear={onClearHistory ?? store.clear}
        />
      )}
    </div>
  );
}
