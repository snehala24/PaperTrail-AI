import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import {
  HISTORY_MAX_ANSWER_CHARS,
  HISTORY_MAX_RUNS,
  HISTORY_STORAGE_KEY,
  buildHistoryEntry,
  clearResearchHistory,
  getResearchHistory,
  historyRunLabel,
  historyToAnswerResponse,
  recordResearchRun,
  removeResearchRun,
  useResearchHistory,
} from "../src/lib/research-history";
import { ResearchHistoryPanel } from "../src/components/history/ResearchHistoryPanel";
import { EvidenceLedgerView } from "../src/components/evidence/EvidenceLedgerView";
import { DashboardView } from "../src/components/dashboard/DashboardView";
import { AnswerResponse } from "../src/lib/types";

// ==========================================
// Fixtures
// ==========================================

const answeredResponse: AnswerResponse = {
  answer: "GFWSVD stabilizes representations through SVD regularization.",
  evidence_links: [
    {
      claim_id: "claim_001",
      chunk_id: "doc_7fdb1200e823_p3_c4",
      relation: "supports",
      rationale: "Directly states the regularization effect.",
      confidence: 0.95,
      doc_id: "doc_7fdb1200e823",
      page: 3,
      quote: "Global Feature-Weighted SVD reduces representation drift.",
    },
  ],
  citation_checks: [
    {
      claim_id: "claim_001",
      chunk_id: "doc_7fdb1200e823_p3_c4",
      exists: true,
      supports: true,
      notes: "Verified.",
      status: "verified",
    },
  ],
  comparisons: [
    {
      claim_id_a: "claim_001",
      claim_id_b: "claim_002",
      doc_id_a: "doc_7fdb1200e823",
      doc_id_b: "doc_c67573ef50cc",
      comparison_type: "supports",
      summary: "Both papers report baseline degradation under shift.",
    },
  ],
  status: "answered",
  timings: { total: 2.4, retrieval: 0.6 },
};

const abstainedResponse: AnswerResponse = {
  answer: "Abstained.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "abstained",
  abstain_reason: "No claim survived citation verification.",
  timings: { total: 1.0 },
};

function entryFor(
  question: string,
  response: AnswerResponse = answeredResponse,
  overrides: Record<string, unknown> = {}
) {
  return buildHistoryEntry({
    question,
    response,
    docIds: null,
    completedAt: "2026-11-02T09:30:00.000Z",
    ...overrides,
  });
}

function Harness() {
  const history = useResearchHistory();
  return (
    <div>
      <span data-testid="count">{history.entries.length}</span>
      <button type="button" onClick={() => history.record({ question: "Recorded?", response: answeredResponse })}>
        record
      </button>
      <button type="button" onClick={history.clear}>
        clear
      </button>
    </div>
  );
}

// ==========================================
// Store tests
// ==========================================

describe("research history store", () => {
  beforeEach(() => {
    localStorage.clear();
    clearResearchHistory();
    vi.restoreAllMocks();
  });

  it("starts empty and keeps nothing until a run is recorded", () => {
    expect(getResearchHistory()).toEqual([]);
    expect(localStorage.getItem(HISTORY_STORAGE_KEY)).toBeNull();
  });

  it("stores the question, answer, citations and verification statuses under a versioned key", () => {
    recordResearchRun(entryFor("How does GFWSVD stabilize representations?"));

    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    expect(raw).toBeTruthy();
    const stored = JSON.parse(raw as string);
    expect(stored).toHaveLength(1);
    expect(stored[0].question).toBe("How does GFWSVD stabilize representations?");
    expect(stored[0].status).toBe("answered");
    expect(stored[0].evidenceLinks[0].quote).toBe(
      "Global Feature-Weighted SVD reduces representation drift."
    );
    expect(stored[0].citationChecks[0].status).toBe("verified");
    expect(stored[0].comparisons).toHaveLength(1);
  });

  it("records newest first and caps the list at the documented maximum", () => {
    for (let i = 0; i < HISTORY_MAX_RUNS + 5; i += 1) {
      recordResearchRun(entryFor(`Question ${i}`));
    }
    const entries = getResearchHistory();
    expect(entries).toHaveLength(HISTORY_MAX_RUNS);
    expect(entries[0].question).toBe(`Question ${HISTORY_MAX_RUNS + 4}`);
    expect(entries[entries.length - 1].question).toBe("Question 5");
  });

  it("round-trips a saved run back into an AnswerResponse the views already understand", () => {
    recordResearchRun(entryFor("Round trip?", answeredResponse));
    const restored = historyToAnswerResponse(getResearchHistory()[0]);
    expect(restored.answer).toBe(answeredResponse.answer);
    expect(restored.evidence_links).toEqual(answeredResponse.evidence_links);
    expect(restored.citation_checks).toEqual(answeredResponse.citation_checks);
    expect(restored.comparisons).toEqual(answeredResponse.comparisons);
    expect(restored.status).toBe("answered");
  });

  it("keeps abstentions as real history without inventing evidence", () => {
    recordResearchRun(entryFor("A question outside the corpus", abstainedResponse));
    const entry = getResearchHistory()[0];
    expect(entry.status).toBe("abstained");
    expect(entry.abstainReason).toBe("No claim survived citation verification.");
    expect(entry.evidenceLinks).toEqual([]);
    expect(historyRunLabel(entry)).toBe("Abstained — no verified evidence");
  });

  it("stores the paper scope when a question was narrowed, and null for the whole corpus", () => {
    recordResearchRun(
      entryFor("Scoped question", answeredResponse, { docIds: ["doc_7fdb1200e823"] })
    );
    recordResearchRun(entryFor("Corpus question", answeredResponse, { docIds: [] }));
    const [corpus, scoped] = getResearchHistory();
    expect(scoped.docIds).toEqual(["doc_7fdb1200e823"]);
    expect(corpus.docIds).toBeNull();
  });

  it("truncates an over-long answer instead of overflowing browser storage", () => {
    const huge: AnswerResponse = {
      ...answeredResponse,
      answer: "x".repeat(HISTORY_MAX_ANSWER_CHARS + 500),
    };
    recordResearchRun(entryFor("Long answer", huge));
    const entry = getResearchHistory()[0];
    expect(entry.answer.length).toBe(HISTORY_MAX_ANSWER_CHARS + 1);
    expect(entry.answer.endsWith("…")).toBe(true);
  });

  it("never persists request timings, credentials or anything outside the answer contract", () => {
    recordResearchRun(entryFor("Hygiene check"));
    const raw = (localStorage.getItem(HISTORY_STORAGE_KEY) ?? "").toLowerCase();
    expect(raw).not.toContain("timings");
    expect(raw).not.toContain("api_key");
    expect(raw).not.toContain("gemini");
    expect(raw).not.toContain("authorization");
    expect(Object.keys(getResearchHistory()[0]).sort()).toEqual(
      [
        "abstainReason",
        "answer",
        "citationChecks",
        "comparisons",
        "completedAt",
        "docIds",
        "evidenceLinks",
        "id",
        "question",
        "status",
        "topK",
      ].sort()
    );
  });

  it("survives corrupt storage by starting clean rather than crashing", () => {
    render(<Harness />);
    localStorage.setItem(HISTORY_STORAGE_KEY, "{not json at all");
    // Same signal another browser tab produces; forces the store to re-read storage.
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", { key: HISTORY_STORAGE_KEY }));
    });
    expect(screen.getByTestId("count").textContent).toBe("0");
  });

  it("drops malformed entries and unknown statuses on load", () => {
    localStorage.setItem(
      HISTORY_STORAGE_KEY,
      JSON.stringify([
        { id: "ok", question: "Good", completedAt: "2026-11-02T09:00:00Z", status: "answered" },
        { question: "No id", status: "answered" },
        { id: "bad-status", question: "Weird", completedAt: "2026-11-02T09:00:00Z", status: "guessed" },
        null,
        "string",
      ])
    );
    render(<Harness />);
    act(() => {
      window.dispatchEvent(new StorageEvent("storage", { key: HISTORY_STORAGE_KEY }));
    });
    expect(screen.getByTestId("count").textContent).toBe("1");
    expect(getResearchHistory()[0].question).toBe("Good");
  });

  it("removes a single run and clears the storage entry when the list empties", () => {
    const a = recordResearchRun(entryFor("Keep me"));
    const b = entryFor("Drop me");
    recordResearchRun(b);
    expect(getResearchHistory()).toHaveLength(2);
    removeResearchRun(b.id);
    expect(getResearchHistory().map((e) => e.id)).toEqual([a.id]);
    removeResearchRun(a.id);
    expect(getResearchHistory()).toEqual([]);
    expect(localStorage.getItem(HISTORY_STORAGE_KEY)).toBeNull();
  });

  it("replaces an existing run that shares an id instead of duplicating it", () => {
    const first = entryFor("Same id", answeredResponse, { id: "dup" });
    recordResearchRun(first);
    recordResearchRun(entryFor("Same id again", abstainedResponse, { id: "dup" }));
    const entries = getResearchHistory();
    expect(entries).toHaveLength(1);
    expect(entries[0].status).toBe("abstained");
  });

  it("publishes updates to mounted consumers through the store subscription", () => {
    render(<Harness />);
    expect(screen.getByTestId("count").textContent).toBe("0");
    fireEvent.click(screen.getByRole("button", { name: "record" }));
    expect(screen.getByTestId("count").textContent).toBe("1");
    fireEvent.click(screen.getByRole("button", { name: "clear" }));
    expect(screen.getByTestId("count").textContent).toBe("0");
  });
});

// ==========================================
// Panel tests
// ==========================================

describe("ResearchHistoryPanel", () => {
  beforeEach(() => {
    localStorage.clear();
    clearResearchHistory();
  });

  it("renders nothing when there is no saved research to show", () => {
    const { container } = render(<ResearchHistoryPanel entries={[]} />);
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText("Saved research")).not.toBeInTheDocument();
  });

  it("lists real runs with their outcome, counts and scope", () => {
    const entry = entryFor("How does GFWSVD stabilize representations?", answeredResponse, {
      docIds: ["doc_7fdb1200e823", "doc_c67573ef50cc"],
    });
    render(<ResearchHistoryPanel entries={[entry]} />);

    expect(screen.getByText("Saved research")).toBeInTheDocument();
    expect(
      screen.getByText(/How does GFWSVD stabilize representations\?/)
    ).toBeInTheDocument();
    expect(screen.getByText("1 claim · 1/1 citations verified")).toBeInTheDocument();
    expect(screen.getByText("2 papers selected")).toBeInTheDocument();
    expect(
      screen.getByText(/Answers stay in this browser/)
    ).toBeInTheDocument();
  });

  it("labels corpus-wide runs and marks the run currently on screen", () => {
    const entry = entryFor("Whole corpus question");
    render(<ResearchHistoryPanel entries={[entry]} activeId={entry.id} />);
    expect(screen.getByText("whole corpus")).toBeInTheDocument();
    expect(screen.getByText("Showing")).toBeInTheDocument();
  });

  it("opens a saved run and removes it on request", () => {
    const onOpen = vi.fn();
    const onDelete = vi.fn();
    const entry = entryFor("Reopen me");
    render(
      <ResearchHistoryPanel
        entries={[entry]}
        onOpen={onOpen}
        onDelete={onDelete}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /Open saved answer for Reopen me/ }));
    expect(onOpen).toHaveBeenCalledWith(entry);

    fireEvent.click(screen.getByRole("button", { name: /Remove saved answer for Reopen me/ }));
    expect(onDelete).toHaveBeenCalledWith(entry.id);
  });

  it("only offers clearing when the caller supports it", () => {
    const entry = entryFor("Question");
    const { rerender } = render(<ResearchHistoryPanel entries={[entry]} />);
    expect(
      screen.queryByRole("button", { name: /Clear all saved research/ })
    ).not.toBeInTheDocument();
    rerender(<ResearchHistoryPanel entries={[entry]} onClear={vi.fn()} />);
    expect(
      screen.getByRole("button", { name: /Clear all saved research/ })
    ).toBeInTheDocument();
  });
});

// ==========================================
// View integration tests
// ==========================================

describe("Evidence Explorer with saved research", () => {
  beforeEach(() => {
    localStorage.clear();
    clearResearchHistory();
  });

  it("shows saved runs even when this session has no answer yet", () => {
    const saved = entryFor("A question from earlier today");
    render(
      <EvidenceLedgerView
        answerResponse={null}
        historyEntries={[saved]}
        onOpenHistoryEntry={vi.fn()}
      />
    );

    expect(screen.getByText("No evidence to show yet")).toBeInTheDocument();
    expect(screen.getByText("Saved research")).toBeInTheDocument();
    expect(screen.getByText(/A question from earlier today/)).toBeInTheDocument();
  });

  it("stays silent about saved research when none exists", () => {
    render(<EvidenceLedgerView answerResponse={answeredResponse} />);
    expect(screen.queryByText("Saved research")).not.toBeInTheDocument();
    expect(
      screen.getByText(/most recent question in your current session/)
    ).toBeInTheDocument();
  });

  it("says so when the ledger is a reopened run rather than a fresh answer", () => {
    const older = entryFor("Older question");
    const newer = entryFor("Newest question");
    render(
      <EvidenceLedgerView
        answerResponse={answeredResponse}
        historyEntries={[newer, older]}
        activeHistoryId={older.id}
      />
    );
    expect(screen.getByText(/Reopened from your saved research/)).toBeInTheDocument();
    expect(
      screen.queryByText(/most recent question in your current session/)
    ).not.toBeInTheDocument();
  });

  it("keeps the newest saved run described as the current result", () => {
    const newest = entryFor("Newest question");
    render(
      <EvidenceLedgerView
        answerResponse={answeredResponse}
        historyEntries={[newest]}
        activeHistoryId={newest.id}
      />
    );
    expect(
      screen.getByText(/most recent question in your current session/)
    ).toBeInTheDocument();
  });

  it("marks a reopened newest run as saved research rather than a fresh question", () => {
    const newest = entryFor("Newest question");
    render(
      <EvidenceLedgerView
        answerResponse={answeredResponse}
        historyEntries={[newest]}
        activeHistoryId={newest.id}
        openedFromHistory
      />
    );
    expect(screen.getByText(/Reopened from your saved research/)).toBeInTheDocument();
    expect(
      screen.queryByText(/most recent question in your current session/)
    ).not.toBeInTheDocument();
  });

  it("states provenance on a reopened abstention without implying fresh verification", () => {
    const prior = entryFor("Earlier question", abstainedResponse);
    render(
      <EvidenceLedgerView
        answerResponse={abstainedResponse}
        historyEntries={[prior]}
        activeHistoryId={prior.id}
        openedFromHistory
      />
    );
    expect(screen.getByText(/Reopened from your saved research/)).toBeInTheDocument();
    expect(screen.getByText("No verified evidence to display")).toBeInTheDocument();
    expect(
      screen.queryByText(/most recent question in your current session/)
    ).not.toBeInTheDocument();
  });

  it("omits the provenance line from an abstention that is still the live result", () => {
    render(
      <EvidenceLedgerView
        answerResponse={abstainedResponse}
        historyEntries={[entryFor("Anything")]}
      />
    );
    expect(
      screen.queryByText(/Reopened from your saved research/)
    ).not.toBeInTheDocument();
  });

  it("clears saved research through the view's callback", () => {
    const onClear = vi.fn();
    render(
      <EvidenceLedgerView
        answerResponse={null}
        historyEntries={[entryFor("Anything")]}
        onClearHistory={onClear}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /Clear all saved research/ }));
    expect(onClear).toHaveBeenCalled();
  });
});

describe("Dashboard with saved research", () => {
  const emptyIndex = () => {
    global.fetch = vi.fn().mockImplementation(async (url: string) => ({
      ok: true,
      status: 200,
      json: async () => (String(url).includes("/chunks") ? [] : []),
    }));
  };

  beforeEach(() => {
    localStorage.clear();
    clearResearchHistory();
    emptyIndex();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows real saved runs and no fabricated activity", async () => {
    const saved = entryFor("What changes under distribution shift?");
    render(
      <DashboardView
        backendStatus="connected"
        onSelectSection={vi.fn()}
        historyEntries={[saved]}
        onOpenHistoryEntry={vi.fn()}
      />
    );

    expect(await screen.findByText("Saved research")).toBeInTheDocument();
    expect(
      screen.getByText(/What changes under distribution shift\?/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/No questions have been run in this session yet/)
    ).toBeInTheDocument();
  });

  it("omits the saved research section entirely when nothing was ever run", async () => {
    render(
      <DashboardView backendStatus="connected" onSelectSection={vi.fn()} historyEntries={[]} />
    );
    await screen.findByText("Answer policy");
    expect(screen.queryByText("Saved research")).not.toBeInTheDocument();
  });
});
