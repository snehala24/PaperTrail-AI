import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { EvidenceLedgerView } from "../src/components/evidence/EvidenceLedgerView";
import {
  AnswerResponse,
  EvidenceLink,
  CitationCheck,
} from "../src/lib/types";

// ==========================================
// Fixtures
// ==========================================

const mockEvidenceLinks: EvidenceLink[] = [
  {
    claim_id: "claim_001",
    chunk_id: "doc_7fdb1200e823_p3_c4",
    relation: "supports",
    rationale: "Directly verifies feature weighting across low-rank SVD.",
    confidence: 0.95,
    doc_id: "doc_7fdb1200e823",
    page: 3,
    quote: "Global Feature-Weighted SVD reduces representation drift.",
  },
  {
    claim_id: "claim_002",
    chunk_id: "doc_c67573ef50cc_p7_c2",
    relation: "weakens",
    rationale: "Shows metric variance under high dimensionality.",
    confidence: 0.82,
    conflict_reason: "Benchmark dataset distributions differ between experiments.",
    doc_id: "doc_c67573ef50cc",
    page: 7,
    quote: "Learned evaluation metrics exhibit domain sensitivity.",
  },
];

const mockCitationChecks: CitationCheck[] = [
  {
    claim_id: "claim_001",
    chunk_id: "doc_7fdb1200e823_p3_c4",
    exists: true,
    supports: true,
    notes: "Layer 1 existence verified; Layer 2/3 entailment confirmed.",
    status: "verified",
  },
  {
    claim_id: "claim_002",
    chunk_id: "doc_c67573ef50cc_p7_c2",
    exists: true,
    supports: false,
    notes: "Chunk discusses sensitivity but does not support generalization.",
    status: "unsupported",
  },
  {
    claim_id: "claim_003",
    chunk_id: "doc_missing_p1_c1",
    exists: false,
    supports: null,
    notes: "Cited chunk not present in retrieved context set.",
    status: "invalid_reference",
  },
];

const mockAnsweredResponse: AnswerResponse = {
  answer: "GFWSVD introduces a global feature-weighting scheme for SVD.",
  evidence_links: mockEvidenceLinks,
  citation_checks: mockCitationChecks,
  comparisons: [],
  status: "answered",
  timings: { total: 2.1 },
};

const mockAbstainedResponse: AnswerResponse = {
  answer: "I must abstain: the evidence could not be verified.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "abstained",
  abstain_reason: "Zero-tolerance: one or more claims failed verification.",
  timings: { total: 1.2 },
};

const mockErrorResponse: AnswerResponse = {
  answer: "An error occurred while generating the answer.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "error",
  timings: { total: 0.1 },
};

// ==========================================
// Tests
// ==========================================

describe("Evidence Explorer view", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ---- Empty state (no answer yet) ----
  describe("No answer state", () => {
    it("renders empty state when answerResponse is null", () => {
      render(<EvidenceLedgerView answerResponse={null} />);
      expect(screen.getByText("No evidence to show yet")).toBeInTheDocument();
      expect(screen.getByText("Evidence Explorer")).toBeInTheDocument();
    });

    it("shows a call to the Research Assistant when callback provided", () => {
      const onGoTo = vi.fn();
      render(<EvidenceLedgerView answerResponse={null} onGoToAssistant={onGoTo} />);
      const btn = screen.getByRole("button", { name: /Ask the Research Assistant/i });
      expect(btn).toBeInTheDocument();
      fireEvent.click(btn);
      expect(onGoTo).toHaveBeenCalledTimes(1);
    });
  });

  // ---- Abstention state ----
  describe("Abstention state", () => {
    it("renders abstention state without fabricating claims", () => {
      render(<EvidenceLedgerView answerResponse={mockAbstainedResponse} />);
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText(/No verified evidence to display/i)).toBeInTheDocument();
      expect(
        screen.getByText(/the ledger is intentionally empty/i)
      ).toBeInTheDocument();
    });

    it("shows the actual abstain reason from backend", () => {
      render(<EvidenceLedgerView answerResponse={mockAbstainedResponse} />);
      expect(
        screen.getByText("Zero-tolerance: one or more claims failed verification.")
      ).toBeInTheDocument();
    });

    it("does NOT render claim ledger cards on abstention", () => {
      render(<EvidenceLedgerView answerResponse={mockAbstainedResponse} />);
      expect(screen.queryByText(/claim_/)).not.toBeInTheDocument();
    });
  });

  // ---- Error state ----
  describe("Error state", () => {
    it("renders error message without fabricating claims", () => {
      render(<EvidenceLedgerView answerResponse={mockErrorResponse} />);
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(
        screen.getByText(/The backend could not complete this request/i)
      ).toBeInTheDocument();
    });
  });

  // ---- Answered state ----
  describe("Answered state", () => {
    it("renders the page header with claim count badge", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(screen.getByText("Evidence Explorer")).toBeInTheDocument();
      expect(
        screen.getByText("2 claims · 1/3 citations verified")
      ).toBeInTheDocument();
    });

    it("renders claim ledger cards for each claim ID", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(screen.getByText("claim_001")).toBeInTheDocument();
      expect(screen.getByText("claim_002")).toBeInTheDocument();
    });

    it("displays correct evidence source count per claim", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      // claim_001 has 1 evidence link
      expect(screen.getAllByText(/1 source passage/i).length).toBeGreaterThan(0);
    });

    it("shows source document ID and page for evidence links", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(screen.getAllByText("doc_7fdb1200e823").length).toBeGreaterThan(0);
      expect(screen.getAllByText("doc_c67573ef50cc").length).toBeGreaterThan(0);
      expect(screen.getAllByText("p. 3").length).toBeGreaterThan(0);
    });

    it("shows verbatim quote from evidence link", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(/Global Feature-Weighted SVD reduces representation drift\./)
      ).toBeInTheDocument();
    });

    it("shows conflict reason when present", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(/Benchmark dataset distributions differ between experiments/i)
      ).toBeInTheDocument();
    });

    it("displays confidence values only when provided by backend", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      // 95% and 82% confidence values
      expect(screen.getByText("95%")).toBeInTheDocument();
      expect(screen.getByText("82%")).toBeInTheDocument();
    });

    it("renders verified verification badge", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(screen.getAllByText("Verified").length).toBeGreaterThan(0);
    });

    it("renders unsupported verification badge", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(screen.getAllByText("Unsupported").length).toBeGreaterThan(0);
    });

    it("explains each verification status in words, not colour alone", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      const legend = screen.getByLabelText("Verification status legend");
      expect(
        within(legend).getByText(
          "The passage exists in the source and its exact text supports the claim."
        )
      ).toBeInTheDocument();
      expect(
        within(legend).getByText(
          "The cited passage is not part of the evidence that was retrieved."
        )
      ).toBeInTheDocument();
    });
  });

  // ---- Citation verification details ----
  describe("Citation verification expansion", () => {
    it("expands citation check details on click showing layer results", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      // The chunk ID appears in both the evidence link body (as a span) and in the citation check
      // row button. Target the citation check row button specifically via its aria-expanded attr.
      const chunkBtns = screen.getAllByText("doc_7fdb1200e823_p3_c4");
      // The citation check row button contains the chunk id in a span inside a button
      // Find the one that's inside a button element
      const checkRowBtn = chunkBtns
        .map((el) => el.closest("button"))
        .find((btn) => btn !== null && btn.hasAttribute("aria-expanded"));
      expect(checkRowBtn).toBeTruthy();
      fireEvent.click(checkRowBtn!);
      // After expanding, auditor notes should appear
      expect(
        screen.getByText("Layer 1 existence verified; Layer 2/3 entailment confirmed.")
      ).toBeInTheDocument();
      expect(screen.getByText(/Passage retrieved/i)).toBeInTheDocument();
      expect(screen.getByText(/Supports claim/i)).toBeInTheDocument();
    });

    it("handles nullable supports field (null → not evaluated)", () => {
      const responseWithNullSupports: AnswerResponse = {
        ...mockAnsweredResponse,
        evidence_links: [
          {
            claim_id: "claim_003",
            chunk_id: "doc_missing_p1_c1",
            relation: "insufficient",
            rationale: "Chunk missing.",
            confidence: 0.0,
            doc_id: null,
            page: null,
            quote: null,
          },
        ],
        citation_checks: [
          {
            claim_id: "claim_003",
            chunk_id: "doc_missing_p1_c1",
            exists: false,
            supports: null,
            notes: "Cited chunk not present in retrieved context set.",
            status: "invalid_reference",
          },
        ],
      };
      render(<EvidenceLedgerView answerResponse={responseWithNullSupports} />);
      // The chunk ID appears in both the evidence link body and the citation check row.
      // Find the citation check row button which has aria-expanded.
      const chunkEls = screen.getAllByText("doc_missing_p1_c1");
      const checkRowBtn = chunkEls
        .map((el) => el.closest("button"))
        .find((btn) => btn !== null && btn.hasAttribute("aria-expanded"));
      expect(checkRowBtn).toBeTruthy();
      fireEvent.click(checkRowBtn!);
      expect(
        screen.getByText("Supports claim").parentElement
      ).toHaveTextContent(/not evaluated/i);
    });
  });

  // ---- Filter behaviour ----
  describe("Evidence filters", () => {
    it("shows filter selects when multiple docs or statuses are available", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      // Two different doc IDs → filter should appear
      const filterSelect = screen.getByLabelText("Filter by document");
      expect(filterSelect).toBeInTheDocument();
    });

    it("filters claims by document ID", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      const docSelect = screen.getByLabelText("Filter by document");
      fireEvent.change(docSelect, { target: { value: "doc_7fdb1200e823" } });
      // claim_001 belongs to doc_7fdb1200e823
      expect(screen.getByText("claim_001")).toBeInTheDocument();
      // claim_002 belongs to doc_c67573ef50cc — should be filtered out
      expect(screen.queryByText("claim_002")).not.toBeInTheDocument();
    });

    it("filters claims by verification status", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      const statusSelect = screen.getByLabelText("Filter by verification status");
      fireEvent.change(statusSelect, { target: { value: "verified" } });
      expect(screen.getByText("claim_001")).toBeInTheDocument();
      expect(screen.queryByText("claim_002")).not.toBeInTheDocument();
    });
  });

  // ---- Current result scope notice ----
  describe("Scope notice", () => {
    it("states that the ledger reflects the latest question in this session", () => {
      render(<EvidenceLedgerView answerResponse={mockAnsweredResponse} />);
      expect(
        screen.getByText(/most recent question in your current session/i)
      ).toBeInTheDocument();
    });

    it("does NOT add session scope text to the abstained state", () => {
      render(<EvidenceLedgerView answerResponse={mockAbstainedResponse} />);
      expect(
        screen.queryByText(/most recent question in your current session/i)
      ).not.toBeInTheDocument();
    });
  });
});
