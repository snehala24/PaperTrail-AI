import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent, within } from "@testing-library/react";
import {
  PdfUpload,
  validatePdfMagicBytes,
} from "../src/components/library/PdfUpload";
import { DocumentCard } from "../src/components/library/DocumentCard";
import { DocumentList } from "../src/components/library/DocumentList";
import { DocumentDetailsModal } from "../src/components/library/DocumentDetailsModal";
import { ChunkExplorerModal } from "../src/components/library/ChunkExplorerModal";
import { ResearchLibraryView } from "../src/components/library/ResearchLibraryView";
import * as api from "../src/lib/api";
import { Document, Chunk, DocumentUploadResponse } from "../src/lib/types";

// ==========================================
// Mock Fixtures
// ==========================================

const mockGfwsvdDoc: Document = {
  doc_id: "doc_7fdb1200e823",
  title: "GFWSVD: Global Feature-Weighted SVD for Multi-Document Cross-Paper Intelligence",
  filename: "2505.17974.pdf",
  sha256: "7fdb1200e823a1b2c3d4e5f67890123456789abcdef0123456789abcdef01234",
  page_count: 19,
  chunker_version: "v2-hierarchical",
  extraction_warnings: [],
};

const mockLemsDoc: Document = {
  doc_id: "doc_c67573ef50cc",
  title: "LEMS: Learned Evaluation Metric Synthesis for Scientific Documents",
  filename: "2610.32878v1.pdf",
  sha256: "c67573ef50cca1b2c3d4e5f67890123456789abcdef0123456789abcdef01234",
  page_count: 34,
  chunker_version: "v2-hierarchical",
  extraction_warnings: ["Page 12: Math symbol glyph fallback to ASCII approximation."],
};

const mockChunks: Chunk[] = [
  {
    chunk_id: "doc_7fdb1200e823_p1_c1",
    doc_id: "doc_7fdb1200e823",
    page: 1,
    section_heading: "Abstract",
    text: "Global Feature-Weighted SVD addresses cross-paper variance in high-dimensional representations.",
    kind: "text",
  },
  {
    chunk_id: "doc_7fdb1200e823_p3_c4",
    doc_id: "doc_7fdb1200e823",
    page: 3,
    section_heading: "3. Methodology & Decomposition",
    text: "Table 1 compares singular value convergence across feature spaces under varying rank cutoffs.",
    kind: "table",
  },
  {
    chunk_id: "doc_7fdb1200e823_p5_c2",
    doc_id: "doc_7fdb1200e823",
    page: 5,
    section_heading: "References",
    text: "[1] Smith et al., Low-rank approximations in neural systems, 2024.",
    kind: "references",
  },
];

/**
 * Creates a mock File object with custom content and mime type.
 */
function createMockPdfFile(name = "test.pdf", size = 1024, magic = "%PDF-1.7"): File {
  const content = new Uint8Array(size);
  // Fill magic bytes
  for (let i = 0; i < magic.length && i < size; i++) {
    content[i] = magic.charCodeAt(i);
  }
  return new File([content], name, { type: "application/pdf" });
}

describe("Research Library & PDF Ingestion", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // Default clipboard mock
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
  // 1. PdfUpload Component Tests
  // ==========================================
  describe("PdfUpload", () => {
    it("validates PDF magic bytes correctly (%PDF-)", async () => {
      const validFile = createMockPdfFile("valid.pdf", 50, "%PDF-");
      const isValid = await validatePdfMagicBytes(validFile);
      expect(isValid).toBe(true);

      const invalidFile = createMockPdfFile("invalid.pdf", 50, "NOTPD");
      const isInvalid = await validatePdfMagicBytes(invalidFile);
      expect(isInvalid).toBe(false);
    });

    it("rejects non-PDF files by extension", async () => {
      render(<PdfUpload />);
      const input = screen.getByLabelText("Upload PDF file input");

      const txtFile = new File(["some text content"], "notes.txt", { type: "text/plain" });
      fireEvent.change(input, { target: { files: [txtFile] } });

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(
          'Only PDF files are supported. Received: "notes.txt"'
        );
      });
    });

    it("rejects files exceeding 25 MB", async () => {
      render(<PdfUpload />);
      const input = screen.getByLabelText("Upload PDF file input");

      // 26 MB file
      const oversizedFile = new File([new Uint8Array(26 * 1024 * 1024)], "heavy.pdf", {
        type: "application/pdf",
      });
      fireEvent.change(input, { target: { files: [oversizedFile] } });

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(
          "File exceeds maximum upload size of 25 MB"
        );
      });
    });

    it("rejects files missing %PDF- magic bytes header", async () => {
      render(<PdfUpload />);
      const input = screen.getByLabelText("Upload PDF file input");

      const fakePdf = createMockPdfFile("corrupt.pdf", 100, "BADHD");
      fireEvent.change(input, { target: { files: [fakePdf] } });

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(
          'Invalid PDF format: "corrupt.pdf" lacks valid %PDF- magic byte header'
        );
      });
    });

    it("successfully selects and uploads valid PDF, triggering callback", async () => {
      const uploadSpy = vi.spyOn(api, "uploadDocument").mockResolvedValueOnce({
        document: mockGfwsvdDoc,
        chunk_count: 78,
      });

      const handleSuccess = vi.fn();
      render(<PdfUpload onUploadSuccess={handleSuccess} />);

      const input = screen.getByLabelText("Upload PDF file input");
      const validPdf = createMockPdfFile("2505.17974.pdf", 2048, "%PDF-");

      fireEvent.change(input, { target: { files: [validPdf] } });

      // File should now be selected and ready
      await waitFor(() => {
        expect(screen.getByText("2505.17974.pdf")).toBeInTheDocument();
        expect(screen.getByText(/Ready for ingestion/i)).toBeInTheDocument();
      });

      // Click upload
      const uploadBtn = screen.getByRole("button", { name: /Upload & Ingest Paper/i });
      fireEvent.click(uploadBtn);

      // Verify upload progress and completion
      await waitFor(() => {
        expect(uploadSpy).toHaveBeenCalledTimes(1);
        expect(handleSuccess).toHaveBeenCalledTimes(1);
        expect(screen.getByRole("status")).toHaveTextContent(
          'Successfully ingested "GFWSVD: Global Feature-Weighted SVD for Multi-Document Cross-Paper Intelligence" (19 pages, 78 chunks indexed).'
        );
      });
    });

    it("handles upload failure with descriptive error alert", async () => {
      vi.spyOn(api, "uploadDocument").mockRejectedValueOnce(
        new api.ApiClientError(413, "File exceeds maximum upload size of 25 MB.")
      );

      render(<PdfUpload />);

      const input = screen.getByLabelText("Upload PDF file input");
      const validPdf = createMockPdfFile("paper.pdf", 2048, "%PDF-");
      fireEvent.change(input, { target: { files: [validPdf] } });

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /Upload & Ingest Paper/i })).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole("button", { name: /Upload & Ingest Paper/i }));

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(
          "File exceeds maximum upload size of 25 MB."
        );
      });

      // Dismiss error
      fireEvent.click(screen.getByLabelText("Dismiss error"));
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
  });

  // ==========================================
  // 2. DocumentCard Component Tests
  // ==========================================
  describe("DocumentCard", () => {
    it("renders document metadata, clean status badge, and copy button", async () => {
      const handleDetails = vi.fn();
      const handleChunks = vi.fn();
      const handleToggle = vi.fn();

      render(
        <DocumentCard
          document={mockGfwsvdDoc}
          chunkCount={78}
          isSelected={false}
          onToggleSelect={handleToggle}
          onViewDetails={handleDetails}
          onInspectChunks={handleChunks}
        />
      );

      expect(screen.getByText(mockGfwsvdDoc.title)).toBeInTheDocument();
      expect(screen.getByText(mockGfwsvdDoc.filename)).toBeInTheDocument();
      expect(screen.getByText("19 pages")).toBeInTheDocument();
      expect(screen.getByText("78 chunks")).toBeInTheDocument();
      expect(screen.getByText("Clean parse")).toBeInTheDocument();

      // Copy doc_id
      const copyBtn = screen.getByLabelText(`Copy document ID ${mockGfwsvdDoc.doc_id}`);
      fireEvent.click(copyBtn);
      await waitFor(() => {
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(mockGfwsvdDoc.doc_id);
      });

      // Actions
      fireEvent.click(screen.getByLabelText(`View metadata details for ${mockGfwsvdDoc.title}`));
      expect(handleDetails).toHaveBeenCalledWith(mockGfwsvdDoc);

      fireEvent.click(screen.getByLabelText(`Inspect chunks for ${mockGfwsvdDoc.title}`));
      expect(handleChunks).toHaveBeenCalledWith(mockGfwsvdDoc);

      // Selection checkbox
      const checkbox = screen.getByLabelText(`Select ${mockGfwsvdDoc.title} for analysis`);
      fireEvent.click(checkbox);
      expect(handleToggle).toHaveBeenCalledWith(mockGfwsvdDoc.doc_id);
    });

    it("renders extraction warnings badge when warnings exist", () => {
      render(
        <DocumentCard
          document={mockLemsDoc}
          onViewDetails={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );

      expect(screen.getByText("1 warning")).toBeInTheDocument();
    });
  });

  // ==========================================
  // 3. DocumentList Component Tests
  // ==========================================
  describe("DocumentList", () => {
    it("renders loading skeletons when loading initial documents", () => {
      render(
        <DocumentList
          documents={[]}
          isLoading={true}
          error={null}
          selectedDocIds={[]}
          onToggleSelect={vi.fn()}
          onRefresh={vi.fn()}
          onViewDetails={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );

      expect(screen.getByText("0 papers")).toBeInTheDocument();
      expect(screen.queryByText("No Research Papers Ingested")).not.toBeInTheDocument();
    });

    it("renders empty state when corpus is empty", () => {
      render(
        <DocumentList
          documents={[]}
          isLoading={false}
          error={null}
          selectedDocIds={[]}
          onToggleSelect={vi.fn()}
          onRefresh={vi.fn()}
          onViewDetails={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );

      expect(screen.getByText("No Research Papers Ingested")).toBeInTheDocument();
    });

    it("renders error alert with retry button", () => {
      const handleRefresh = vi.fn();
      render(
        <DocumentList
          documents={[]}
          isLoading={false}
          error="Network error: connection refused"
          selectedDocIds={[]}
          onToggleSelect={vi.fn()}
          onRefresh={handleRefresh}
          onViewDetails={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );

      expect(screen.getByRole("alert")).toHaveTextContent("Network error: connection refused");
      fireEvent.click(screen.getByRole("button", { name: /Retry/i }));
      expect(handleRefresh).toHaveBeenCalledTimes(1);
    });

    it("renders list of documents and filters by search query", () => {
      render(
        <DocumentList
          documents={[mockGfwsvdDoc, mockLemsDoc]}
          isLoading={false}
          error={null}
          selectedDocIds={[]}
          onToggleSelect={vi.fn()}
          onSelectAll={vi.fn()}
          onClearSelection={vi.fn()}
          onRefresh={vi.fn()}
          onViewDetails={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );

      expect(screen.getByText("2 papers")).toBeInTheDocument();
      expect(screen.getByText("53 total extracted pages across corpus")).toBeInTheDocument();
      expect(screen.getByText(mockGfwsvdDoc.title)).toBeInTheDocument();
      expect(screen.getByText(mockLemsDoc.title)).toBeInTheDocument();

      // Search for GFWSVD
      const searchInput = screen.getByLabelText("Filter documents by query");
      fireEvent.change(searchInput, { target: { value: "GFWSVD" } });

      expect(screen.getByText(mockGfwsvdDoc.title)).toBeInTheDocument();
      expect(screen.queryByText(mockLemsDoc.title)).not.toBeInTheDocument();

      // Search for non-existent paper
      fireEvent.change(searchInput, { target: { value: "QuantumSupercomputer" } });
      expect(screen.getByText("No matching papers")).toBeInTheDocument();

      // Clear filter
      fireEvent.click(screen.getByRole("button", { name: "Clear filter" }));
      expect(screen.getByText(mockGfwsvdDoc.title)).toBeInTheDocument();
      expect(screen.getByText(mockLemsDoc.title)).toBeInTheDocument();
    });
  });

  // ==========================================
  // 4. DocumentDetailsModal Component Tests
  // ==========================================
  describe("DocumentDetailsModal", () => {
    it("renders document metadata, SHA-256 digest, and extraction diagnostics", async () => {
      const handleClose = vi.fn();
      const handleInspectChunks = vi.fn();

      render(
        <DocumentDetailsModal
          document={mockLemsDoc}
          isOpen={true}
          onClose={handleClose}
          onInspectChunks={handleInspectChunks}
        />
      );

      expect(screen.getByText("Document Metadata & Diagnostics")).toBeInTheDocument();
      expect(screen.getByText(mockLemsDoc.title)).toBeInTheDocument();
      expect(screen.getByText(mockLemsDoc.doc_id)).toBeInTheDocument();
      expect(screen.getByText(mockLemsDoc.filename)).toBeInTheDocument();
      expect(screen.getByText("34 pages")).toBeInTheDocument();
      expect(screen.getByText(mockLemsDoc.sha256)).toBeInTheDocument();

      // Warning listed
      expect(screen.getByText(/1 Parser Warning Recorded/i)).toBeInTheDocument();
      expect(screen.getByText(/Math symbol glyph fallback/i)).toBeInTheDocument();

      // Copy SHA-256
      fireEvent.click(screen.getByLabelText("Copy SHA-256 hash"));
      await waitFor(() => {
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(mockLemsDoc.sha256);
      });

      // Click inspect chunks
      fireEvent.click(screen.getByRole("button", { name: /Inspect Chunks \(34 pages\)/i }));
      expect(handleClose).toHaveBeenCalledTimes(1);
      expect(handleInspectChunks).toHaveBeenCalledWith(mockLemsDoc);
    });

    it("explains a legacy figure warning without rewriting the stored wording", () => {
      render(
        <DocumentDetailsModal
          document={{
            ...mockGfwsvdDoc,
            extraction_warnings: [
              "1 pages contain image blocks that were not extracted: pages 6.",
            ],
          }}
          isOpen={true}
          onClose={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );

      // The recorded warning is shown exactly as stored.
      expect(
        screen.getByText("1 pages contain image blocks that were not extracted: pages 6.")
      ).toBeInTheDocument();
      // Plus a plain-language note on what was and was not extracted.
      expect(
        screen.getByText(/still contribute their body text, table contents and figure captions/i)
      ).toBeInTheDocument();
    });

    it("does not add the figure note to unrelated warnings", () => {
      render(
        <DocumentDetailsModal
          document={mockLemsDoc}
          isOpen={true}
          onClose={vi.fn()}
          onInspectChunks={vi.fn()}
        />
      );
      expect(
        screen.queryByText(/still contribute their body text/i)
      ).not.toBeInTheDocument();
    });

    it("dismisses on Escape key press", () => {
      const handleClose = vi.fn();
      render(
        <DocumentDetailsModal
          document={mockGfwsvdDoc}
          isOpen={true}
          onClose={handleClose}
          onInspectChunks={vi.fn()}
        />
      );

      fireEvent.keyDown(window, { key: "Escape" });
      expect(handleClose).toHaveBeenCalledTimes(1);
    });
  });

  // ==========================================
  // 5. ChunkExplorerModal Component Tests
  // ==========================================
  describe("ChunkExplorerModal", () => {
    it("fetches chunks from backend and renders chunk cards with filters", async () => {
      const fetchSpy = vi.spyOn(api, "getDocumentChunks").mockResolvedValueOnce(mockChunks);
      const handleClose = vi.fn();

      render(
        <ChunkExplorerModal
          document={mockGfwsvdDoc}
          isOpen={true}
          onClose={handleClose}
        />
      );

      // Loading state
      expect(screen.getByText("Retrieving granular chunks from backend...")).toBeInTheDocument();

      // Once loaded
      await waitFor(() => {
        expect(fetchSpy).toHaveBeenCalledWith(mockGfwsvdDoc.doc_id);
        expect(screen.getByLabelText("Chunk count summary")).toHaveTextContent("Showing 3 of 3 chunks");
      });

      // Verify chunk cards
      expect(screen.getByText("doc_7fdb1200e823_p1_c1")).toBeInTheDocument();
      expect(screen.getAllByText("Page 1").length).toBeGreaterThan(0);
      expect(screen.getByText("Abstract")).toBeInTheDocument();

      expect(screen.getByText("doc_7fdb1200e823_p3_c4")).toBeInTheDocument();
      expect(screen.getAllByText("Page 3").length).toBeGreaterThan(0);
      expect(screen.getByText("3. Methodology & Decomposition")).toBeInTheDocument();

      // Filter by semantic kind "table"
      const kindSelect = screen.getByLabelText("Filter by semantic kind");
      fireEvent.change(kindSelect, { target: { value: "table" } });

      await waitFor(() => {
        expect(screen.getByLabelText("Chunk count summary")).toHaveTextContent("Showing 1 of 3 chunks");
        expect(screen.getByText("doc_7fdb1200e823_p3_c4")).toBeInTheDocument();
        expect(screen.queryByText("doc_7fdb1200e823_p1_c1")).not.toBeInTheDocument();
      });

      // Filter by keyword in text
      fireEvent.change(kindSelect, { target: { value: "all" } });
      const searchInput = screen.getByLabelText("Search chunks text or section heading");
      fireEvent.change(searchInput, { target: { value: "neural systems" } });

      await waitFor(() => {
        expect(screen.getByLabelText("Chunk count summary")).toHaveTextContent("Showing 1 of 3 chunks");
        expect(screen.getByText("doc_7fdb1200e823_p5_c2")).toBeInTheDocument();
      });

      // Copy chunk ID
      fireEvent.click(screen.getByLabelText("Copy chunk ID doc_7fdb1200e823_p5_c2"));
      await waitFor(() => {
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith("doc_7fdb1200e823_p5_c2");
      });
    });

    it("displays error banner when chunk retrieval fails", async () => {
      vi.spyOn(api, "getDocumentChunks").mockRejectedValueOnce(
        new api.ApiClientError(404, "Document with id 'doc_7fdb1200e823' not found.")
      );

      render(
        <ChunkExplorerModal
          document={mockGfwsvdDoc}
          isOpen={true}
          onClose={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole("alert")).toHaveTextContent(
          "Document with id 'doc_7fdb1200e823' not found."
        );
      });
    });
  });

  // ==========================================
  // 6. ResearchLibraryView Integration Tests
  // ==========================================
  describe("ResearchLibraryView", () => {
    it("loads documents on mount and displays complete library view", async () => {
      vi.spyOn(api, "listDocuments").mockResolvedValueOnce([mockGfwsvdDoc, mockLemsDoc]);

      render(<ResearchLibraryView />);

      await waitFor(() => {
        expect(screen.getByText("Research Library")).toBeInTheDocument();
        expect(screen.getByText("2 papers in the index")).toBeInTheDocument();
        expect(screen.getByText(mockGfwsvdDoc.title)).toBeInTheDocument();
        expect(screen.getByText(mockLemsDoc.title)).toBeInTheDocument();
      });
    });

    it("opens document details modal and inspect chunks modal through interactive flow", async () => {
      vi.spyOn(api, "listDocuments").mockResolvedValueOnce([mockGfwsvdDoc]);
      vi.spyOn(api, "getDocumentChunks").mockResolvedValueOnce(mockChunks);

      render(<ResearchLibraryView />);

      await waitFor(() => {
        expect(screen.getByText(mockGfwsvdDoc.title)).toBeInTheDocument();
      });

      // Open details modal
      const detailsBtn = screen.getByLabelText(`View metadata details for ${mockGfwsvdDoc.title}`);
      fireEvent.click(detailsBtn);

      expect(screen.getByText("Document Metadata & Diagnostics")).toBeInTheDocument();
      expect(screen.getByText(mockGfwsvdDoc.sha256)).toBeInTheDocument();

      // From details modal, click inspect chunks
      const inspectBtn = screen.getByRole("button", {
        name: /Inspect Chunks \(19 pages\)/i,
      });
      fireEvent.click(inspectBtn);

      // Details modal closes, Chunk Explorer opens
      await waitFor(() => {
        expect(screen.getByText("Granular Chunk Explorer")).toBeInTheDocument();
        expect(screen.getByText("doc_7fdb1200e823_p1_c1")).toBeInTheDocument();
      });

      // Close chunk explorer
      fireEvent.click(screen.getByLabelText("Close chunk explorer"));
      await waitFor(() => {
        expect(screen.queryByText("Granular Chunk Explorer")).not.toBeInTheDocument();
      });
    });
  });
});
