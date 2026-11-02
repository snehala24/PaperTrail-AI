"use client";

/**
 * Local research history.
 *
 * Stores the answers this browser has already produced so a researcher can come
 * back to a previous question and re-open its evidence ledger without re-running
 * the pipeline.
 *
 * Deliberate constraints:
 * - Browser-local only. Nothing is sent anywhere and nothing is written to the backend.
 * - Only question text, answer text, citations and verification statuses are kept.
 *   No credentials, no request payloads, no environment values.
 * - Bounded in size and capped in count so a long session cannot overflow storage.
 * - Every read is defensive: a corrupt or missing entry never breaks the workspace.
 */

import { useSyncExternalStore } from "react";
import {
  AnswerResponse,
  AnswerStatus,
  ClaimComparison,
  CitationCheck,
  EvidenceLink,
} from "@/lib/types";

export const HISTORY_STORAGE_KEY = "papertrail.research-history.v1";
export const HISTORY_MAX_RUNS = 20;
export const HISTORY_MAX_ANSWER_CHARS = 4000;
const HISTORY_MAX_LIST_ITEMS = 60;

export interface ResearchHistoryEntry {
  /** Stable local id for this run. */
  id: string;
  /** The question exactly as it was asked. */
  question: string;
  /** ISO timestamp captured when the answer arrived. */
  completedAt: string;
  /** Answer outcome. */
  status: AnswerStatus;
  /** Papers the question was scoped to; null means the whole corpus. */
  docIds: string[] | null;
  /** Evidence depth that was requested, if any. */
  topK: number | null;
  /** Answer or abstention text produced by the backend. */
  answer: string;
  /** Abstention reason, when the answer was withheld. */
  abstainReason: string | null;
  evidenceLinks: EvidenceLink[];
  citationChecks: CitationCheck[];
  comparisons: ClaimComparison[];
}

export interface BuildEntryInput {
  question: string;
  response: AnswerResponse;
  docIds?: string[] | null;
  topK?: number | null;
  id?: string;
  completedAt?: string;
}

const EMPTY_HISTORY: ResearchHistoryEntry[] = [];

let cache: ResearchHistoryEntry[] | null = null;
const listeners = new Set<() => void>();

function newId(): string {
  const cryptoObj = typeof crypto !== "undefined" ? crypto : undefined;
  if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
    return cryptoObj.randomUUID();
  }
  return `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function storage(): Storage | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage ?? null;
  } catch {
    return null;
  }
}

function truncate(value: string, max: number): string {
  if (typeof value !== "string") return "";
  return value.length > max ? `${value.slice(0, max)}…` : value;
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]).slice(0, HISTORY_MAX_LIST_ITEMS) : [];
}

/** Only accepts entries that still look like a recorded run; anything else is dropped. */
function normalizeEntry(value: unknown): ResearchHistoryEntry | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Partial<ResearchHistoryEntry>;
  if (typeof raw.id !== "string" || !raw.id) return null;
  if (typeof raw.question !== "string") return null;
  if (raw.status !== "answered" && raw.status !== "abstained" && raw.status !== "error") {
    return null;
  }
  if (typeof raw.completedAt !== "string") return null;

  return {
    id: raw.id,
    question: truncate(raw.question, 500),
    completedAt: raw.completedAt,
    status: raw.status,
    docIds: Array.isArray(raw.docIds) ? raw.docIds.filter((d) => typeof d === "string") : null,
    topK: typeof raw.topK === "number" ? raw.topK : null,
    answer: truncate(typeof raw.answer === "string" ? raw.answer : "", HISTORY_MAX_ANSWER_CHARS),
    abstainReason: typeof raw.abstainReason === "string" ? raw.abstainReason : null,
    evidenceLinks: asArray<EvidenceLink>(raw.evidenceLinks),
    citationChecks: asArray<CitationCheck>(raw.citationChecks),
    comparisons: asArray<ClaimComparison>(raw.comparisons),
  };
}

function readFromStorage(): ResearchHistoryEntry[] {
  const store = storage();
  if (!store) return EMPTY_HISTORY;
  let raw: string | null = null;
  try {
    raw = store.getItem(HISTORY_STORAGE_KEY);
  } catch {
    return EMPTY_HISTORY;
  }
  if (!raw) return EMPTY_HISTORY;

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return EMPTY_HISTORY;
    const entries = parsed
      .map(normalizeEntry)
      .filter((e): e is ResearchHistoryEntry => e !== null)
      .slice(0, HISTORY_MAX_RUNS);
    return entries.length > 0 ? entries : EMPTY_HISTORY;
  } catch {
    return EMPTY_HISTORY;
  }
}

function writeToStorage(entries: ResearchHistoryEntry[]): void {
  const store = storage();
  if (!store) return;
  try {
    if (entries.length === 0) {
      store.removeItem(HISTORY_STORAGE_KEY);
      return;
    }
    store.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Quota or privacy-mode failure: the session keeps working, history just stops growing.
  }
}

function emit(): void {
  listeners.forEach((listener) => {
    try {
      listener();
    } catch {
      // A broken subscriber must not stop the others from updating.
    }
  });
}

export function buildHistoryEntry(input: BuildEntryInput): ResearchHistoryEntry {
  const { response } = input;
  return {
    id: input.id ?? newId(),
    question: truncate(input.question ?? "", 500),
    completedAt: input.completedAt ?? new Date().toISOString(),
    status: response.status,
    docIds:
      input.docIds && input.docIds.length > 0
        ? input.docIds.filter((d) => typeof d === "string")
        : null,
    topK: typeof input.topK === "number" ? input.topK : null,
    answer: truncate(response.answer ?? "", HISTORY_MAX_ANSWER_CHARS),
    abstainReason:
      typeof response.abstain_reason === "string" ? response.abstain_reason : null,
    evidenceLinks: (response.evidence_links ?? []).slice(0, HISTORY_MAX_LIST_ITEMS),
    citationChecks: (response.citation_checks ?? []).slice(0, HISTORY_MAX_LIST_ITEMS),
    comparisons: (response.comparisons ?? []).slice(0, HISTORY_MAX_LIST_ITEMS),
  };
}

export function historyToAnswerResponse(entry: ResearchHistoryEntry): AnswerResponse {
  return {
    answer: entry.answer,
    evidence_links: entry.evidenceLinks,
    citation_checks: entry.citationChecks,
    comparisons: entry.comparisons,
    status: entry.status,
    abstain_reason: entry.abstainReason,
    timings: {},
  };
}

/** Newest first, capped, and always a stable reference for React consumers. */
export function getResearchHistory(): ResearchHistoryEntry[] {
  if (cache === null) cache = readFromStorage();
  return cache;
}

function getServerSnapshot(): ResearchHistoryEntry[] {
  return EMPTY_HISTORY;
}

export function replaceResearchHistory(
  entries: ResearchHistoryEntry[]
): ResearchHistoryEntry[] {
  cache = entries.slice(0, HISTORY_MAX_RUNS);
  writeToStorage(cache);
  emit();
  return cache;
}

/** Adds a run to the front of the list and drops the oldest beyond the cap. */
export function recordResearchRun(entry: ResearchHistoryEntry): ResearchHistoryEntry {
  const existing = getResearchHistory().filter((e) => e.id !== entry.id);
  replaceResearchHistory([entry, ...existing]);
  return entry;
}

export function removeResearchRun(id: string): ResearchHistoryEntry[] {
  return replaceResearchHistory(getResearchHistory().filter((e) => e.id !== id));
}

export function clearResearchHistory(): ResearchHistoryEntry[] {
  return replaceResearchHistory([]);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Cross-tab awareness: another tab writing history refreshes this one.
 * Also resets the cache so a read always reflects storage.
 */
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event: StorageEvent) => {
    if (event.key !== HISTORY_STORAGE_KEY && event.key !== null) return;
    cache = readFromStorage();
    emit();
  });
}

export interface ResearchHistoryApi {
  entries: ResearchHistoryEntry[];
  record: (input: BuildEntryInput) => ResearchHistoryEntry;
  remove: (id: string) => void;
  clear: () => void;
}

export function useResearchHistory(): ResearchHistoryApi {
  const entries = useSyncExternalStore(
    subscribe,
    getResearchHistory,
    getServerSnapshot
  );

  const record = (input: BuildEntryInput) => {
    const entry = buildHistoryEntry(input);
    recordResearchRun(entry);
    return entry;
  };

  return {
    entries,
    record,
    remove: (id: string) => {
      removeResearchRun(id);
    },
    clear: () => {
      clearResearchHistory();
    },
  };
}

export function formatHistoryTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "Time unavailable";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function historyRunLabel(entry: ResearchHistoryEntry): string {
  const verified = entry.citationChecks.filter((c) => c.status === "verified").length;
  const claims = new Set(entry.evidenceLinks.map((l) => l.claim_id)).size;
  if (entry.status === "answered") {
    return `${claims} claim${claims === 1 ? "" : "s"} · ${verified}/${entry.citationChecks.length} citations verified`;
  }
  if (entry.status === "abstained") return "Abstained — no verified evidence";
  return "Request failed";
}
