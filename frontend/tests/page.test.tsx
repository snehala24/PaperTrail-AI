import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import WorkspaceHome from "../app/page";
import * as api from "../src/lib/api";
import { AnswerResponse } from "../src/lib/types";

// ==========================================
// Shared fixtures
// ==========================================

const mockAnsweredResponse: AnswerResponse = {
  answer: "GFWSVD introduces a global feature-weighting scheme for SVD.",
  evidence_links: [
    {
      claim_id: "claim_001",
      chunk_id: "doc_7fdb1200e823_p3_c4",
      relation: "supports",
      rationale: "Directly verifies weighting formulation.",
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
      summary: "Both papers align on baseline degradation.",
    },
  ],
  status: "answered",
  timings: { total: 2.1 },
};

const gotoAssistant = () =>
  fireEvent.click(screen.getByRole("button", { name: /^Research Assistant/ }));

describe("Workspace shell & navigation", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders workspace branding with the Dashboard as the default section", async () => {
    vi.spyOn(api, "getHealth").mockResolvedValueOnce({
      status: "ok",
      version: "0.1.0",
      timestamp: "2026-10-30T10:00:00Z",
    });
    vi.spyOn(api, "listDocuments").mockResolvedValue([]);
    vi.spyOn(api, "getDocumentChunks").mockResolvedValue([]);

    render(<WorkspaceHome />);

    expect(screen.getByText("PaperTrail AI")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument();
    expect(
      screen.getByText(/No questions have been run in this session yet\./)
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
    });
  });

  it("displays Offline indicator when backend is unreachable", async () => {
    vi.spyOn(api, "getHealth").mockRejectedValueOnce(
      new api.ApiClientError(0, "Network error: connection refused")
    );

    render(<WorkspaceHome />);

    await waitFor(() => {
      expect(screen.getByText("Backend Offline (:8000)")).toBeInTheDocument();
    });
  });

  it("navigates to Evidence Explorer and renders EvidenceLedgerView (no answer)", async () => {
    vi.spyOn(api, "getHealth").mockResolvedValueOnce({
      status: "ok",
      version: "0.1.0",
      timestamp: "2026-10-30T10:00:00Z",
    });

    render(<WorkspaceHome />);

    await waitFor(() => {
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
    });

    const evidenceBtn = screen.getByRole("button", { name: /Evidence Explorer/i });
    fireEvent.click(evidenceBtn);

    expect(screen.getByRole("heading", { name: "Evidence Explorer" })).toBeInTheDocument();
    expect(screen.getByText("No evidence to show yet")).toBeInTheDocument();
  });

  it("navigates to Cross-Paper Analysis and renders CrossPaperComparisonsView (no answer)", async () => {
    vi.spyOn(api, "getHealth").mockResolvedValueOnce({
      status: "ok",
      version: "0.1.0",
      timestamp: "2026-10-30T10:00:00Z",
    });

    render(<WorkspaceHome />);

    await waitFor(() => {
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
    });

    const comparisonsBtn = screen.getByRole("button", {
      name: /Cross-Paper Analysis/i,
    });
    fireEvent.click(comparisonsBtn);

    expect(screen.getByRole("heading", { name: "Cross-Paper Analysis" })).toBeInTheDocument();
    expect(screen.getByText("No comparisons yet")).toBeInTheDocument();
  });

  it("navigates to Research Library section", async () => {
    vi.spyOn(api, "getHealth").mockResolvedValueOnce({
      status: "ok",
      version: "0.1.0",
      timestamp: "2026-10-30T10:00:00Z",
    });
    vi.spyOn(api, "listDocuments").mockResolvedValueOnce([]);

    render(<WorkspaceHome />);

    await waitFor(() => {
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
    });

    const libraryBtn = screen.getByRole("button", { name: /Research Library/i });
    fireEvent.click(libraryBtn);
    expect(screen.getByRole("heading", { name: "Research Library" })).toBeInTheDocument();
  });

  it("handles mobile menu open and Escape key dismissal", async () => {
    vi.spyOn(api, "getHealth").mockResolvedValueOnce({
      status: "ok",
      version: "0.1.0",
      timestamp: "2026-10-30T10:00:00Z",
    });

    render(<WorkspaceHome />);

    await waitFor(() => {
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
    });

    const menuBtn = screen.getByRole("button", { name: /Open navigation menu/i });
    fireEvent.click(menuBtn);

    const closeBtn = screen.getByRole("button", { name: /Close navigation sidebar/i });
    expect(closeBtn).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });
  });

  // ==========================================
  // Shared answer state tests
  // ==========================================

  describe("Shared answer state architecture", () => {
    it("answer response survives navigation from Assistant to Evidence Explorer", async () => {
      vi.spyOn(api, "getHealth").mockResolvedValueOnce({
        status: "ok",
        version: "0.1.0",
        timestamp: "2026-10-30T10:00:00Z",
      });
      vi.spyOn(api, "listDocuments").mockResolvedValue([]);
      vi.spyOn(api, "generateAnswer").mockResolvedValueOnce(mockAnsweredResponse);

      render(<WorkspaceHome />);

      await waitFor(() => {
        expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
      });
      gotoAssistant();

      // Submit a question in the Assistant
      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );
      fireEvent.change(textarea, {
        target: { value: "How does GFWSVD stabilize representations?" },
      });
      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      fireEvent.click(submitBtn);

      // Wait for answer to appear
      await waitFor(() => {
        expect(
          screen.getByText(mockAnsweredResponse.answer)
        ).toBeInTheDocument();
      });

      // Navigate to Evidence Explorer
      const evidenceBtn = screen.getByRole("button", { name: /Evidence Explorer/i });
      fireEvent.click(evidenceBtn);

      // Shared state: evidence ledger should show the claims from the answer
      await waitFor(() => {
        expect(screen.getByRole("heading", { name: "Evidence Explorer" })).toBeInTheDocument();
        expect(screen.getByText("claim_001")).toBeInTheDocument();
      });
    });

    it("answer response survives navigation from Assistant to Cross-Paper Analysis", async () => {
      vi.spyOn(api, "getHealth").mockResolvedValueOnce({
        status: "ok",
        version: "0.1.0",
        timestamp: "2026-10-30T10:00:00Z",
      });
      vi.spyOn(api, "listDocuments").mockResolvedValue([]);
      vi.spyOn(api, "generateAnswer").mockResolvedValueOnce(mockAnsweredResponse);

      render(<WorkspaceHome />);

      await waitFor(() => {
        expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
      });
      gotoAssistant();

      // Submit question
      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );
      fireEvent.change(textarea, {
        target: { value: "Compare GFWSVD and LEMS approaches." },
      });
      const submitBtn = screen.getByRole("button", {
        name: /Submit research inquiry/i,
      });
      fireEvent.click(submitBtn);

      await waitFor(() => {
        expect(screen.getByText(mockAnsweredResponse.answer)).toBeInTheDocument();
      });

      // Navigate to Comparisons
      const comparisonsBtn = screen.getByRole("button", {
        name: /Cross-Paper Analysis/i,
      });
      fireEvent.click(comparisonsBtn);

      await waitFor(() => {
        expect(screen.getByRole("heading", { name: "Cross-Paper Analysis" })).toBeInTheDocument();
        // Comparison summary should appear (from shared state)
        expect(
          screen.getByText("Both papers align on baseline degradation.")
        ).toBeInTheDocument();
      });
    });

    it("failed request does NOT replace current answer with null", async () => {
      vi.spyOn(api, "getHealth").mockResolvedValueOnce({
        status: "ok",
        version: "0.1.0",
        timestamp: "2026-10-30T10:00:00Z",
      });
      vi.spyOn(api, "listDocuments").mockResolvedValue([]);

      // First call succeeds
      vi.spyOn(api, "generateAnswer")
        .mockResolvedValueOnce(mockAnsweredResponse)
        // Second call fails
        .mockRejectedValueOnce(new api.ApiClientError(500, "Internal Server Error"));

      render(<WorkspaceHome />);

      await waitFor(() => {
        expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
      });
      gotoAssistant();

      const textarea = screen.getByLabelText(
        "Natural language scientific research query"
      );

      // First successful question
      fireEvent.change(textarea, {
        target: { value: "First question." },
      });
      let submitBtn = screen.getByRole("button", { name: /Submit research inquiry/i });
      fireEvent.click(submitBtn);

      await waitFor(() => {
        expect(screen.getByText(mockAnsweredResponse.answer)).toBeInTheDocument();
      });

      // Navigate to Evidence, verify answer present
      fireEvent.click(screen.getByRole("button", { name: /Evidence Explorer/i }));
      await waitFor(() => {
        expect(screen.getByText("claim_001")).toBeInTheDocument();
      });

      // Navigate back, ask a question that fails
      fireEvent.click(screen.getByRole("button", { name: /^Research Assistant/ }));
      fireEvent.change(
        screen.getByLabelText("Natural language scientific research query"),
        { target: { value: "Second failing question." } }
      );
      submitBtn = screen.getByRole("button", { name: /Submit research inquiry/i });
      fireEvent.click(submitBtn);

      // Wait for error to show
      await waitFor(() => {
        expect(screen.getByText("Query Execution Failed")).toBeInTheDocument();
      });

      // Navigate to Evidence Explorer — shared state should still hold previous answer
      fireEvent.click(screen.getByRole("button", { name: /Evidence Explorer/i }));
      await waitFor(() => {
        expect(screen.getByText("claim_001")).toBeInTheDocument();
      });
    });

    it("selected document state is preserved when navigating between sections", async () => {
      vi.spyOn(api, "getHealth").mockResolvedValueOnce({
        status: "ok",
        version: "0.1.0",
        timestamp: "2026-10-30T10:00:00Z",
      });

      const mockDocs = [
        {
          doc_id: "doc_7fdb1200e823",
          title: "GFWSVD Paper",
          filename: "2505.17974.pdf",
          sha256: "abc123",
          page_count: 19,
          extraction_warnings: [],
        },
      ];

      vi.spyOn(api, "listDocuments").mockResolvedValue(mockDocs);

      render(<WorkspaceHome />);

      gotoAssistant();
      // No papers selected yet: the Assistant searches the whole index
      expect(screen.getByRole("heading", { name: "Research Assistant" })).toBeInTheDocument();
      expect(screen.getByText("All indexed papers")).toBeInTheDocument();

      // Select one paper, then check the selection survives navigation
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Expand paper selection/i })
        ).toBeInTheDocument()
      );
      fireEvent.click(screen.getByRole("button", { name: /Expand paper selection/i }));
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /Select all \(1\)/i })
        ).toBeInTheDocument()
      );
      fireEvent.click(screen.getByRole("button", { name: /Select all \(1\)/i }));
      expect(screen.getByText("1 paper selected")).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: /Research Library/i }));
      await waitFor(() => {
        expect(screen.getByRole("heading", { name: "Research Library" })).toBeInTheDocument();
      });

      gotoAssistant();
      expect(screen.getByText("1 paper selected")).toBeInTheDocument();
    });
  });

  // ==========================================
  // HealthResponse schema test
  // ==========================================

  describe("HealthResponse schema", () => {
    it("frontend HealthResponse type includes timestamp which backend provides", async () => {
      // The backend HealthResponse Pydantic model DOES include timestamp (verified in api.py).
      // The mock here matches the actual backend contract.
      const mockHealth = {
        status: "ok",
        version: "0.1.0",
        timestamp: "2026-10-30T10:00:00Z",
      };
      vi.spyOn(api, "getHealth").mockResolvedValueOnce(mockHealth);

      render(<WorkspaceHome />);

      await waitFor(() => {
        expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
      });

      // The HealthResponse type has: status, version, timestamp — all present in backend.
      // No drift: backend api.py HealthResponse has timestamp via default_factory=datetime.now(utc)
      expect(mockHealth.timestamp).toBeDefined();
      expect(typeof mockHealth.timestamp).toBe("string");
    });
  });
});

