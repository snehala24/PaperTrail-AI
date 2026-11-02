import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CrossPaperComparisonsView } from "../src/components/comparisons/CrossPaperComparisonsView";
import { AnswerResponse, ClaimComparison } from "../src/lib/types";

// ==========================================
// Fixtures
// ==========================================

const mockComparisons: ClaimComparison[] = [
  {
    claim_id_a: "claim_001",
    claim_id_b: "claim_002",
    doc_id_a: "doc_7fdb1200e823",
    doc_id_b: "doc_c67573ef50cc",
    comparison_type: "apparent_conflict",
    summary:
      "GFWSVD maintains stability via SVD regularization, while LEMS observes metric sensitivity.",
    conflict_reason:
      "GFWSVD evaluates on static embeddings; LEMS evaluates dynamically synthesized test prompts.",
  },
  {
    claim_id_a: "claim_003",
    claim_id_b: "claim_004",
    doc_id_a: "doc_7fdb1200e823",
    doc_id_b: "doc_c67573ef50cc",
    comparison_type: "supports",
    summary:
      "Both methods demonstrate that unweighted baselines degrade under distribution shifts.",
  },
  {
    claim_id_a: "claim_005",
    claim_id_b: "claim_006",
    doc_id_a: "doc_7fdb1200e823",
    doc_id_b: "doc_c67573ef50cc",
    comparison_type: "contradicts",
    summary: "Direct empirical contradiction in reported compute latency on identical GPU hardware.",
  },
  {
    claim_id_a: "claim_007",
    claim_id_b: "claim_008",
    doc_id_a: "doc_7fdb1200e823",
    doc_id_b: "doc_c67573ef50cc",
    comparison_type: "unresolved",
    summary:
      "Cannot resolve convergence properties due to differing stopping criteria between implementations.",
  },
];

const mockAnsweredResponse: AnswerResponse = {
  answer: "Grounded answer text.",
  evidence_links: [],
  citation_checks: [],
  comparisons: mockComparisons,
  status: "answered",
  timings: { total: 2.5 },
};

const mockAbstainedResponse: AnswerResponse = {
  answer: "Abstained.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "abstained",
  abstain_reason: "Zero-tolerance policy triggered.",
  timings: { total: 1.1 },
};

const mockNoComparisonsResponse: AnswerResponse = {
  answer: "Answer with no comparisons.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "answered",
  timings: { total: 0.9 },
};

const mockErrorResponse: AnswerResponse = {
  answer: "Backend error occurred.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "error",
  timings: { total: 0.1 },
};

// ==========================================
// Tests
// ==========================================

describe("Cross-Paper Analysis view", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ---- No answer state ----
  describe("No answer state", () => {
    it("renders empty state when answerResponse is null", () => {
      render(<CrossPaperComparisonsView answerResponse={null} />);
      expect(screen.getByText("No comparisons yet")).toBeInTheDocument();
      expect(screen.getByText("Cross-Paper Analysis")).toBeInTheDocument();
    });

    it("shows Go to Research Assistant button when callback provided", () => {
      const onGoTo = vi.fn();
      render(
        <CrossPaperComparisonsView answerResponse={null} onGoToAssistant={onGoTo} />
      );
      const btn = screen.getByRole("button", { name: /Ask the Research Assistant/i });
      expect(btn).toBeInTheDocument();
      fireEvent.click(btn);
      expect(onGoTo).toHaveBeenCalledTimes(1);
    });
  });

  // ---- Abstention state ----
  describe("Abstention state", () => {
    it("shows abstained state without fabricating comparisons", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAbstainedResponse} />);
      expect(screen.getByRole("alert")).toBeInTheDocument();
      // h3 heading specifically
      expect(
        screen.getByRole("heading", { name: /Nothing to compare/i })
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Comparisons are never generated from unverified claims/i)
      ).toBeInTheDocument();
    });

    it("does NOT render comparison cards on abstention", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAbstainedResponse} />);
      expect(screen.queryByText(/claim_001/)).not.toBeInTheDocument();
    });
  });

  // ---- Error state ----
  describe("Error state", () => {
    it("renders error message without fabricating comparisons", () => {
      render(<CrossPaperComparisonsView answerResponse={mockErrorResponse} />);
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(
        screen.getByText(/The backend could not complete this request/i)
      ).toBeInTheDocument();
    });
  });

  // ---- Empty comparisons on answered response ----
  describe("Answered with no comparisons", () => {
    it("renders no-comparisons state when comparisons array is empty", () => {
      render(<CrossPaperComparisonsView answerResponse={mockNoComparisonsResponse} />);
      expect(
        screen.getByText("This answer did not span multiple papers")
      ).toBeInTheDocument();
    });
  });

  // ---- Answered with comparisons ----
  describe("All four comparison types", () => {
    it("renders all four comparison relationship labels", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(screen.getByText("Apparent Conflict")).toBeInTheDocument();
      expect(screen.getByText("Supports / Harmonious")).toBeInTheDocument();
      expect(screen.getByText("Contradiction")).toBeInTheDocument();
      expect(screen.getByText("Unresolved / Divergent Context")).toBeInTheDocument();
    });

    it("preserves Claim A and Claim B identifiers", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(screen.getAllByText("claim_001").length).toBeGreaterThan(0);
      expect(screen.getAllByText("claim_002").length).toBeGreaterThan(0);
    });

    it("preserves source document attributions for both sides", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      // Every card names the source paper of side A and side B. With no document
      // catalogue in this unit test, PaperRef degrades to the raw doc_id — it never
      // invents a title.
      expect(screen.getAllByText("doc_7fdb1200e823").length).toBeGreaterThanOrEqual(4);
      expect(screen.getAllByText("doc_c67573ef50cc").length).toBeGreaterThanOrEqual(4);
    });

    it("displays the synthesis summary for each comparison", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(
          "GFWSVD maintains stability via SVD regularization, while LEMS observes metric sensitivity."
        )
      ).toBeInTheDocument();
    });
  });

  // ---- Conflict reason ----
  describe("Apparent conflict reason", () => {
    it("displays conflict reason when provided by backend", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(
          "GFWSVD evaluates on static embeddings; LEMS evaluates dynamically synthesized test prompts."
        )
      ).toBeInTheDocument();
    });

    it("does NOT display conflict reason section when not provided", () => {
      // The 'supports' comparison has no conflict_reason
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      // The "Source of the difference" header should only appear once (for the apparent_conflict entry)
      const conflictHeaders = screen.queryAllByText("Source of the difference");
      expect(conflictHeaders.length).toBe(1);
    });
  });

  // ---- Classification integrity ----
  describe("Classification integrity", () => {
    it("does NOT misclassify unresolved as contradicts", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      // unresolved comparison text
      expect(
        screen.getByText(
          "Cannot resolve convergence properties due to differing stopping criteria between implementations."
        )
      ).toBeInTheDocument();
      // The classification shown should be "Unresolved / Divergent Context" — not "Contradiction"
      expect(screen.getByText("Unresolved / Divergent Context")).toBeInTheDocument();
    });

    it("does NOT misclassify apparent_conflict as contradicts", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      // apparent_conflict label should appear
      expect(screen.getByText("Apparent Conflict")).toBeInTheDocument();
      // Description must say the difference is methodological, not factual.
      expect(
        screen.getAllByText(/different methods, assumptions, datasets or settings/i)
          .length
      ).toBeGreaterThan(1);
      expect(
        screen.getAllByText(/not by a factual disagreement/i).length
      ).toBeGreaterThan(0);
    });

    it("shows conservative unresolved note for unresolved comparisons", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(/Left unresolved deliberately/)
      ).toBeInTheDocument();
    });
  });

  // ---- Filter by type (chip buttons) ----
  describe("Filter by comparison type", () => {
    it("filters comparisons when type chip is clicked", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      // Click the "Supports (1)" chip to filter
      const supportsChip = screen.getByRole("button", { name: /Supports \(1\)/i });
      fireEvent.click(supportsChip);
      // Only the supports comparison should show
      expect(
        screen.getByText(
          "Both methods demonstrate that unweighted baselines degrade under distribution shifts."
        )
      ).toBeInTheDocument();
      // Apparent conflict comparison should be hidden
      expect(
        screen.queryByText(
          "GFWSVD maintains stability via SVD regularization, while LEMS observes metric sensitivity."
        )
      ).not.toBeInTheDocument();
    });

    it("clears filter when same chip is clicked again", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      const supportsChip = screen.getByRole("button", { name: /Supports \(1\)/i });
      fireEvent.click(supportsChip); // filter on
      fireEvent.click(supportsChip); // filter off
      // All comparisons should be visible again
      expect(screen.getByText("Apparent Conflict")).toBeInTheDocument();
      expect(screen.getByText("Supports / Harmonious")).toBeInTheDocument();
    });
  });

  // ---- Scope notice ----
  describe("Scope notice", () => {
    it("states that findings come from the latest question in this session", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(/most recent question in your current session/i)
      ).toBeInTheDocument();
    });

    it("does NOT show scope notice on abstained response", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAbstainedResponse} />);
      expect(
        screen.queryByText(/most recent question in your current session/i)
      ).not.toBeInTheDocument();
    });
  });

  // ---- Source evidence behind both sides ----
  describe("Source evidence behind each side", () => {
    const withQuotes: AnswerResponse = {
      ...mockAnsweredResponse,
      comparisons: [mockComparisons[0]],
      evidence_links: [
        {
          claim_id: "claim_001",
          chunk_id: "doc_7fdb1200e823_p4_c2",
          relation: "supports",
          rationale: "States the stability result.",
          confidence: 1,
          doc_id: "doc_7fdb1200e823",
          page: 4,
          quote: "GFWSVD maintains stability through regularized decomposition.",
        },
        {
          claim_id: "claim_002",
          chunk_id: "doc_c67573ef50cc_p6_c5",
          relation: "supports",
          rationale: "States the degradation result.",
          confidence: 1,
          doc_id: "doc_c67573ef50cc",
          page: 6,
          quote: "GFWSVD degrades significantly under identical conditions.",
        },
      ],
    };

    it("shows the exact quoted passage behind both sides of a finding", () => {
      render(<CrossPaperComparisonsView answerResponse={withQuotes} />);
      expect(
        screen.getByText(
          "“GFWSVD maintains stability through regularized decomposition.”"
        )
      ).toBeInTheDocument();
      expect(
        screen.getByText("“GFWSVD degrades significantly under identical conditions.”")
      ).toBeInTheDocument();
      expect(screen.getByText(/p\. 4 · doc_7fdb1200e823_p4_c2/)).toBeInTheDocument();
      expect(screen.getByText(/p\. 6 · doc_c67573ef50cc_p6_c5/)).toBeInTheDocument();
    });

    it("says honestly when a claim has no recorded passage", () => {
      render(<CrossPaperComparisonsView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getAllByText(/No quoted passage was captured in the answer/i).length
      ).toBeGreaterThan(0);
    });

    it("marks findings that were reopened from saved research", () => {
      render(
        <CrossPaperComparisonsView
          answerResponse={withQuotes}
          sourceNote="Reopened from your saved research on Nov 2, 04:00 PM."
        />
      );
      expect(
        screen.getByText(/Reopened from your saved research on Nov 2, 04:00 PM\./)
      ).toBeInTheDocument();
      expect(
        screen.queryByText(/most recent question in your current session/i)
      ).not.toBeInTheDocument();
    });
  });
});
