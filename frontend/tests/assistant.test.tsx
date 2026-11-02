import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TopKControl } from "../src/components/assistant/TopKControl";
import { DocumentSelector } from "../src/components/assistant/DocumentSelector";
import { QuestionComposer } from "../src/components/assistant/QuestionComposer";
import { ClaimCard } from "../src/components/assistant/ClaimCard";
import { CitationVerification } from "../src/components/assistant/CitationVerification";
import { ComparisonPanel } from "../src/components/assistant/ComparisonPanel";
import { AbstentionState } from "../src/components/assistant/AbstentionState";
import { AnswerDiagnostics } from "../src/components/assistant/AnswerDiagnostics";
import { AnswerPanel } from "../src/components/assistant/AnswerPanel";
import { ResearchAssistantView } from "../src/components/assistant/ResearchAssistantView";
import * as api from "../src/lib/api";
import {
  Document,
  AnswerResponse,
  EvidenceLink,
  CitationCheck,
  ClaimComparison,
} from "../src/lib/types";

// ==========================================
// Mock Fixtures
// ==========================================

const mockDocs: Document[] = [
  {
    doc_id: "doc_7fdb1200e823",
    title: "GFWSVD: Global Feature-Weighted SVD",
    filename: "2505.17974.pdf",
    sha256: "7fdb1200e823a1b2c3d4e5f67890123456789abcdef0123456789abcdef01234",
    page_count: 19,
    chunker_version: "v2-hierarchical",
    extraction_warnings: [],
  },
  {
    doc_id: "doc_c67573ef50cc",
    title: "LEMS: Learned Evaluation Metric Synthesis",
    filename: "2610.32878v1.pdf",
    sha256: "c67573ef50cca1b2c3d4e5f67890123456789abcdef0123456789abcdef01234",
    page_count: 34,
    chunker_version: "v2-hierarchical",
    extraction_warnings: [],
  },
];

const mockEvidenceLinks: EvidenceLink[] = [
  {
    claim_id: "claim_001",
    chunk_id: "doc_7fdb1200e823_p3_c4",
    relation: "supports",
    rationale: "Directly verifies feature weighting formulation across low-rank singular values.",
    confidence: 0.95,
    doc_id: "doc_7fdb1200e823",
    page: 3,
    quote: "Global Feature-Weighted SVD reduces representation drift across disparate papers.",
  },
  {
    claim_id: "claim_002",
    chunk_id: "doc_c67573ef50cc_p7_c2",
    relation: "weakens",
    rationale: "Shows metric variance under high dimensionality.",
    confidence: 0.82,
    conflict_reason: "Differences in benchmark dataset sample distributions",
    doc_id: "doc_c67573ef50cc",
    page: 7,
    quote: "Learned evaluation metrics exhibit domain sensitivity when out-of-distribution.",
  },
];

const mockCitationChecks: CitationCheck[] = [
  {
    claim_id: "claim_001",
    chunk_id: "doc_7fdb1200e823_p3_c4",
    exists: true,
    supports: true,
    notes: "Layer 1 existence verified; Layer 2/3 semantic entailment confirmed.",
    status: "verified",
  },
  {
    claim_id: "claim_002",
    chunk_id: "doc_c67573ef50cc_p7_c2",
    exists: true,
    supports: false,
    notes: "Chunk discusses domain sensitivity but does not strictly support the generalization claim.",
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

const mockComparisons: ClaimComparison[] = [
  {
    claim_id_a: "claim_001",
    claim_id_b: "claim_002",
    doc_id_a: "doc_7fdb1200e823",
    doc_id_b: "doc_c67573ef50cc",
    comparison_type: "apparent_conflict",
    summary: "GFWSVD maintains stability through SVD regularization, whereas LEMS observes sensitivity due to metric learning.",
    conflict_reason: "GFWSVD evaluates on static embeddings, while LEMS evaluates dynamically synthesized test prompts.",
  },
  {
    claim_id_a: "claim_003",
    claim_id_b: "claim_004",
    doc_id_a: "doc_7fdb1200e823",
    doc_id_b: "doc_c67573ef50cc",
    comparison_type: "supports",
    summary: "Both methods demonstrate that unweighted baselines degrade under cross-domain distribution shifts.",
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
    summary: "Cannot resolve convergence properties due to differing stopping criteria between implementations.",
  },
];

const mockAnsweredResponse: AnswerResponse = {
  answer: "GFWSVD introduces a global feature-weighting scheme for singular value decomposition that mitigates representation collapse across multi-paper corpora.",
  evidence_links: mockEvidenceLinks,
  citation_checks: mockCitationChecks,
  comparisons: mockComparisons,
  status: "answered",
  timings: {
    retrieval: 0.045,
    rerank: 0.120,
    generation: 1.450,
    verification: 0.820,
    total: 2.435,
  },
};

const mockAbstainedResponse: AnswerResponse = {
  answer: "I must abstain because the available evidence does not establish the singular value threshold used in GFWSVD experiments.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "abstained",
  abstain_reason: "Available evidence in doc_7fdb1200e823 does not mention a rank-50 threshold; Layer 3 verification rejected unverified extrapolation.",
  timings: {
    retrieval: 0.040,
    generation: 1.100,
    verification: 0.600,
    total: 1.740,
  },
};

const mockErrorResponse: AnswerResponse = {
  answer: "An error occurred while generating the grounded answer: Gemini daily quota exhausted.",
  evidence_links: [],
  citation_checks: [],
  comparisons: [],
  status: "error",
  abstain_reason: "LLM generation aborted: Gemini daily quota exhausted (free-tier daily limit reached).",
  timings: {
    total: 0.150,
  },
};

describe("Grounded Research Assistant", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ==========================================
  // 1. TopKControl Tests
  // ==========================================
  describe("TopKControl", () => {
    it("renders slider and number input with initial value", () => {
      render(<TopKControl value={8} onChange={vi.fn()} />);
      expect(screen.getByLabelText("Current top_k value: 8")).toBeInTheDocument();
      const slider = screen.getByRole("slider");
      expect(slider).toHaveValue("8");
    });

    it("clamps values between 1 and 10", () => {
      const handleChange = vi.fn();
      render(<TopKControl value={8} onChange={handleChange} />);

      const numberInput = screen.getByLabelText("Direct top_k number input");
      fireEvent.change(numberInput, { target: { value: "15" } });
      expect(handleChange).toHaveBeenCalledWith(10);

      fireEvent.change(numberInput, { target: { value: "-5" } });
      expect(handleChange).toHaveBeenCalledWith(1);
    });
  });

  // ==========================================
  // 2. DocumentSelector Tests
  // ==========================================
  describe("DocumentSelector", () => {
    it("renders Entire Corpus state when no papers are selected", () => {
      render(
        <DocumentSelector
          documents={mockDocs}
          selectedDocIds={[]}
          onToggleDoc={vi.fn()}
          onSelectAll={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      expect(screen.getByText("Entire Corpus (All Papers)")).toBeInTheDocument();
      expect(screen.getByText("Unfiltered")).toBeInTheDocument();
    });

    it("renders Filter Applied state with paper count and chips when selected", () => {
      render(
        <DocumentSelector
          documents={mockDocs}
          selectedDocIds={["doc_7fdb1200e823"]}
          onToggleDoc={vi.fn()}
          onSelectAll={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      expect(
        screen.getByText("Filtered Corpus (1 of 2 papers)")
      ).toBeInTheDocument();
      expect(screen.getByText("Filter Applied")).toBeInTheDocument();
      expect(screen.getByText(mockDocs[0].title)).toBeInTheDocument();
    });

    it("allows toggling paper checkboxes in expanded dropdown", () => {
      const handleToggle = vi.fn();
      render(
        <DocumentSelector
          documents={mockDocs}
          selectedDocIds={["doc_7fdb1200e823"]}
          onToggleDoc={handleToggle}
          onSelectAll={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      // Expand dropdown
      const expandBtn = screen.getByLabelText("Expand paper selection");
      fireEvent.click(expandBtn);

      const checkbox = screen.getByLabelText(`Select paper ${mockDocs[1].title}`);
      fireEvent.click(checkbox);
      expect(handleToggle).toHaveBeenCalledWith(mockDocs[1].doc_id);
    });
  });

  // ==========================================
  // 3. QuestionComposer Tests
  // ==========================================
  describe("QuestionComposer", () => {
    it("disables submit button when question input is empty or whitespace", () => {
      render(
        <QuestionComposer
          question=""
          onChangeQuestion={vi.fn()}
          onSubmit={vi.fn()}
          isLoading={false}
          topK={8}
          onChangeTopK={vi.fn()}
          documents={mockDocs}
          selectedDocIds={[]}
          onToggleDoc={vi.fn()}
          onSelectAllDocs={vi.fn()}
          onClearAllDocs={vi.fn()}
        />
      );

      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      expect(submitBtn).toBeDisabled();
    });

    it("submits on button click and keyboard shortcut (Ctrl+Enter)", () => {
      const handleSubmit = vi.fn();
      render(
        <QuestionComposer
          question="How does GFWSVD feature weighting operate?"
          onChangeQuestion={vi.fn()}
          onSubmit={handleSubmit}
          isLoading={false}
          topK={8}
          onChangeTopK={vi.fn()}
          documents={mockDocs}
          selectedDocIds={[]}
          onToggleDoc={vi.fn()}
          onSelectAllDocs={vi.fn()}
          onClearAllDocs={vi.fn()}
        />
      );

      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      expect(submitBtn).not.toBeDisabled();
      fireEvent.click(submitBtn);
      expect(handleSubmit).toHaveBeenCalledTimes(1);

      // Keyboard shortcut Ctrl+Enter
      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );
      fireEvent.keyDown(textarea, { key: "Enter", ctrlKey: true });
      expect(handleSubmit).toHaveBeenCalledTimes(2);
    });

    it("prevents duplicate submissions while loading", () => {
      const handleSubmit = vi.fn();
      render(
        <QuestionComposer
          question="Comparing SVD methods"
          onChangeQuestion={vi.fn()}
          onSubmit={handleSubmit}
          isLoading={true}
          topK={8}
          onChangeTopK={vi.fn()}
          documents={mockDocs}
          selectedDocIds={[]}
          onToggleDoc={vi.fn()}
          onSelectAllDocs={vi.fn()}
          onClearAllDocs={vi.fn()}
        />
      );

      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      expect(submitBtn).toBeDisabled();
      expect(screen.getByText("Synthesizing & Verifying...")).toBeInTheDocument();
    });
  });

  // ==========================================
  // 4. CitationVerification Component Tests
  // ==========================================
  describe("CitationVerification", () => {
    it("renders 3-layer audit items with appropriate status badges", () => {
      render(<CitationVerification checks={mockCitationChecks} />);

      expect(
        screen.getByText("3-Layer Citation Verification Audit")
      ).toBeInTheDocument();

      // Check verified status
      expect(screen.getByText("Verified")).toBeInTheDocument();
      expect(screen.getAllByText("Retrieved in Context").length).toBeGreaterThan(0);
      expect(screen.getByText("Entails Claim")).toBeInTheDocument();

      // Check unsupported status
      expect(screen.getByText("Unsupported")).toBeInTheDocument();
      expect(screen.getByText("Fails Entailment")).toBeInTheDocument();

      // Check invalid reference status
      expect(screen.getByText("Invalid Reference")).toBeInTheDocument();
      expect(screen.getByText("Missing from Evidence Set")).toBeInTheDocument();
    });
  });

  // ==========================================
  // 5. ComparisonPanel Component Tests
  // ==========================================
  describe("ComparisonPanel", () => {
    it("renders all four comparison types and preserves source attribution", () => {
      render(<ComparisonPanel comparisons={mockComparisons} />);

      expect(
        screen.getByText("Cross-Paper Comparative Synthesis (4)")
      ).toBeInTheDocument();

      // All 4 types
      expect(screen.getByText("Apparent Conflict")).toBeInTheDocument();
      expect(screen.getByText("Supports / Harmonious")).toBeInTheDocument();
      expect(screen.getByText("Contradiction")).toBeInTheDocument();
      expect(
        screen.getByText("Unresolved / Divergent Context")
      ).toBeInTheDocument();

      // Apparent conflict reason
      expect(
        screen.getByText(/GFWSVD evaluates on static embeddings/i)
      ).toBeInTheDocument();

      // Source attributions: with no document catalogue mounted, PaperRef shows the
      // raw doc_id rather than inventing a paper title.
      expect(screen.getAllByText("doc_7fdb1200e823").length).toBe(4);
      expect(screen.getAllByText("doc_c67573ef50cc").length).toBe(4);
    });
  });

  // ==========================================
  // 6. AbstentionState Tests
  // ==========================================
  describe("AbstentionState", () => {
    it("displays honest fail-closed abstention explanation and reason", () => {
      render(
        <AbstentionState
          reason="Available evidence in doc_7fdb1200e823 does not establish singular value cutoff."
        />
      );

      expect(
        screen.getByText("System Abstained: Insufficient or Unverified Evidence")
      ).toBeInTheDocument();
      expect(
        screen.getByText(
          "Available evidence in doc_7fdb1200e823 does not establish singular value cutoff."
        )
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Fail-Closed Guarantee: No synthetic claims/i)
      ).toBeInTheDocument();
    });
  });

  // ==========================================
  // 7. AnswerPanel Tests
  // ==========================================
  describe("AnswerPanel", () => {
    it("renders loading state with detailed pipeline message", () => {
      render(
        <AnswerPanel
          response={null}
          isLoading={true}
          error={null}
          onReset={vi.fn()}
        />
      );

      expect(
        screen.getByText("Synthesizing & Verifying Grounded Answer")
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Searching the selected papers/i)
      ).toBeInTheDocument();
    });

    it("renders error state when network or client error occurs", () => {
      render(
        <AnswerPanel
          response={null}
          isLoading={false}
          error="Network error: Connection refused"
          onReset={vi.fn()}
        />
      );

      expect(screen.getByText("Query Execution Failed")).toBeInTheDocument();
      expect(
        screen.getByText("Network error: Connection refused")
      ).toBeInTheDocument();
    });

    it("renders backend error status distinctly from scientific abstention", () => {
      render(
        <AnswerPanel
          response={mockErrorResponse}
          isLoading={false}
          error={null}
          onReset={vi.fn()}
        />
      );

      expect(screen.getByText("Backend Execution Error")).toBeInTheDocument();
      expect(screen.getByText("Internal Engine Error")).toBeInTheDocument();
      expect(
        screen.getAllByText(/Gemini daily quota exhausted/i).length
      ).toBeGreaterThan(0);
    });

    it("renders scientific abstention without fabricated claims or citations", () => {
      render(
        <AnswerPanel
          response={mockAbstainedResponse}
          isLoading={false}
          error={null}
          onReset={vi.fn()}
        />
      );

      expect(
        screen.getByText("System Abstained: Insufficient or Unverified Evidence")
      ).toBeInTheDocument();
      // Ensure no claims or evidence cards are rendered
      expect(
        screen.queryByText(/Grounded Claims & Verbatim Evidence/i)
      ).not.toBeInTheDocument();
    });

    it("renders answered response with answer text, claim cards, verbatim quote, and copy action", async () => {
      render(
        <AnswerPanel
          response={mockAnsweredResponse}
          isLoading={false}
          error={null}
          onReset={vi.fn()}
        />
      );

      expect(screen.getByText("Evidence-Grounded Answer")).toBeInTheDocument();
      expect(screen.getByText(mockAnsweredResponse.answer)).toBeInTheDocument();
      expect(
        screen.getByText("Grounded Claims & Verbatim Evidence (2)")
      ).toBeInTheDocument();
      expect(screen.getAllByText("claim_001").length).toBeGreaterThan(0);
      expect(screen.getAllByText("doc_7fdb1200e823_p3_c4").length).toBeGreaterThan(0);
      expect(
        screen.getByText(/Global Feature-Weighted SVD reduces representation drift/i)
      ).toBeInTheDocument();

      // Copy answer button
      const copyBtn = screen.getByLabelText("Copy answer text");
      fireEvent.click(copyBtn);
      await waitFor(() => {
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
          mockAnsweredResponse.answer
        );
      });
    });
  });

  // ==========================================
  // 8. ResearchAssistantView Integration Tests
  // ==========================================
  describe("ResearchAssistantView", () => {
    it("submits research query to backend and renders answered response", async () => {
      vi.spyOn(api, "listDocuments").mockResolvedValueOnce(mockDocs);
      const answerSpy = vi
        .spyOn(api, "generateAnswer")
        .mockResolvedValueOnce(mockAnsweredResponse);

      const handleDocChange = vi.fn();
      const handleAnswerGenerated = vi.fn();
      render(
        <ResearchAssistantView
          selectedDocIds={["doc_7fdb1200e823"]}
          onSelectedDocIdsChange={handleDocChange}
          onAnswerGenerated={handleAnswerGenerated}
        />
      );

      // Wait for documents to load
      await waitFor(() => {
        expect(screen.getByText("1 paper selected")).toBeInTheDocument();
      });

      // Type question
      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );
      fireEvent.change(textarea, {
        target: {
          value:
            "Explain how GFWSVD feature weighting stabilizes representations.",
        },
      });

      // Submit
      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      fireEvent.click(submitBtn);

      // Verify API was called with exact parameters
      await waitFor(() => {
        expect(answerSpy).toHaveBeenCalledWith({
          question:
            "Explain how GFWSVD feature weighting stabilizes representations.",
          doc_ids: ["doc_7fdb1200e823"],
          top_k: 8,
        });
        expect(
          screen.getByText(mockAnsweredResponse.answer)
        ).toBeInTheDocument();
        // onAnswerGenerated called with the response
        expect(handleAnswerGenerated).toHaveBeenCalledWith(
            mockAnsweredResponse,
            "Explain how GFWSVD feature weighting stabilizes representations."
          );
      });
    });

    it("submits research query with entire corpus when no doc_ids are selected", async () => {
      vi.spyOn(api, "listDocuments").mockResolvedValueOnce(mockDocs);
      const answerSpy = vi
        .spyOn(api, "generateAnswer")
        .mockResolvedValueOnce(mockAbstainedResponse);

      render(
        <ResearchAssistantView
          selectedDocIds={[]}
          onSelectedDocIdsChange={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByText("All indexed papers")).toBeInTheDocument();
      });

      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );
      fireEvent.change(textarea, {
        target: {
          value: "What is the rank-50 threshold in GFWSVD?",
        },
      });

      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      fireEvent.click(submitBtn);

      await waitFor(() => {
        expect(answerSpy).toHaveBeenCalledWith({
          question: "What is the rank-50 threshold in GFWSVD?",
          doc_ids: undefined,
          top_k: 8,
        });
        expect(
          screen.getByText(
            "System Abstained: Insufficient or Unverified Evidence"
          )
        ).toBeInTheDocument();
      });
    });

    it("onAnswerGenerated is optional and omitting it does not throw", async () => {
      vi.spyOn(api, "listDocuments").mockResolvedValueOnce(mockDocs);
      vi.spyOn(api, "generateAnswer").mockResolvedValueOnce(mockAnsweredResponse);

      // No onAnswerGenerated prop — should work without errors
      render(
        <ResearchAssistantView
          selectedDocIds={[]}
          onSelectedDocIdsChange={vi.fn()}
        />
      );

      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );
      fireEvent.change(textarea, {
        target: { value: "Optional callback test." },
      });
      fireEvent.click(
        screen.getByRole("button", { name: /Submit research inquiry/i })
      );

      await waitFor(() => {
        expect(
          screen.getByText(mockAnsweredResponse.answer)
        ).toBeInTheDocument();
      });
    });
  });
});
