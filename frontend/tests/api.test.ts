import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  getApiBaseUrl,
  getHealth,
  listDocuments,
  getDocument,
  getDocumentChunks,
  searchCorpus,
  generateAnswer,
  uploadDocument,
  ApiClientError,
} from "../src/lib/api";
import {
  HealthResponse,
  Document,
  Chunk,
  AnswerResponse,
  SearchResponse,
  DocumentUploadResponse,
} from "../src/lib/types";

describe("API Client & Contracts (Phase 1)", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  describe("Base URL Configuration", () => {
    it("defaults to http://localhost:8000 when NEXT_PUBLIC_API_URL is unset", () => {
      delete process.env.NEXT_PUBLIC_API_URL;
      expect(getApiBaseUrl()).toBe("http://localhost:8000");
    });

    it("uses NEXT_PUBLIC_API_URL when set, stripping trailing slashes", () => {
      process.env.NEXT_PUBLIC_API_URL = "https://api.papertrail.ai///";
      expect(getApiBaseUrl()).toBe("https://api.papertrail.ai");
    });
  });

  describe("Endpoint Requests with Mock Responses", () => {
    it("getHealth returns valid HealthResponse", async () => {
      const mockHealth: HealthResponse = {
        status: "ok",
        version: "0.1.0",
        timestamp: "2026-10-30T10:00:00Z",
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealth,
      });

      const res = await getHealth();
      expect(res.status).toBe("ok");
      expect(res.version).toBe("0.1.0");
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:8000/health",
        expect.objectContaining({ method: "GET" })
      );
    });

    it("listDocuments returns array of Document contracts", async () => {
      const mockDocs: Document[] = [
        {
          doc_id: "doc_c67573ef50cc",
          title: "LEMS Paper",
          filename: "2610.32878v1.pdf",
          sha256: "c67573ef50ccf236151da342e53ba1319560146f4ccdd104e6cb41ad24439922",
          page_count: 34,
          chunker_version: "0.2.3",
          extraction_warnings: [],
        },
        {
          doc_id: "doc_7fdb1200e823",
          title: "GFWSVD Paper",
          filename: "2505.17974.pdf",
          sha256: "7fdb1200e8234a0596be542431eecb8a2caefdbf94a32e22aaf34f948d7d2eab",
          page_count: 19,
          chunker_version: "0.2.3",
          extraction_warnings: ["Page 6 image omitted"],
        },
      ];

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockDocs,
      });

      const docs = await listDocuments();
      expect(docs).toHaveLength(2);
      expect(docs[0].doc_id).toBe("doc_c67573ef50cc");
      expect(docs[1].page_count).toBe(19);
    });

    it("getDocument retrieves single document metadata", async () => {
      const mockDoc: Document = {
        doc_id: "doc_c67573ef50cc",
        title: "LEMS Paper",
        filename: "2610.32878v1.pdf",
        sha256: "c67573ef50ccf236151da342e53ba1319560146f4ccdd104e6cb41ad24439922",
        page_count: 34,
        extraction_warnings: [],
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockDoc,
      });

      const doc = await getDocument("doc_c67573ef50cc");
      expect(doc.doc_id).toBe("doc_c67573ef50cc");
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:8000/documents/doc_c67573ef50cc",
        expect.anything()
      );
    });

    it("getDocumentChunks retrieves chunk array", async () => {
      const mockChunks: Chunk[] = [
        {
          chunk_id: "doc_c67573ef50cc_p3_c5",
          doc_id: "doc_c67573ef50cc",
          page: 3,
          section_heading: "3. Methodology",
          text: "Kronecker-factor estimation via Lanczos",
          kind: "text",
        },
      ];

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockChunks,
      });

      const chunks = await getDocumentChunks("doc_c67573ef50cc");
      expect(chunks).toHaveLength(1);
      expect(chunks[0].chunk_id).toBe("doc_c67573ef50cc_p3_c5");
    });

    it("searchCorpus submits search payload and returns SearchResponse", async () => {
      const mockSearch: SearchResponse = {
        query: "Kronecker factor",
        results: [],
        total_candidates: 45,
        returned_count: 0,
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockSearch,
      });

      const res = await searchCorpus({ query: "Kronecker factor", top_k: 5 });
      expect(res.total_candidates).toBe(45);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:8000/search",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ query: "Kronecker factor", top_k: 5 }),
        })
      );
    });

    it("generateAnswer handles answered response", async () => {
      const mockAnswer: AnswerResponse = {
        answer: "LEMS and GFWSVD differ in factor estimation.",
        evidence_links: [
          {
            claim_id: "claim_1",
            chunk_id: "doc_7fdb1200e823_p5_c5",
            relation: "supports",
            rationale: "Explicit description",
            confidence: 1.0,
            conflict_reason: null,
            doc_id: "doc_7fdb1200e823",
            page: 5,
            quote: "per-sample gradients on calibration dataset",
          },
        ],
        citation_checks: [
          {
            claim_id: "claim_1",
            chunk_id: "doc_7fdb1200e823_p5_c5",
            exists: true,
            supports: true,
            notes: "Verified",
            status: "verified",
          },
        ],
        comparisons: [
          {
            claim_id_a: "claim_1",
            claim_id_b: "claim_2",
            doc_id_a: "doc_7fdb1200e823",
            doc_id_b: "doc_c67573ef50cc",
            comparison_type: "unresolved",
            summary: "Different processes",
            conflict_reason: null,
          },
        ],
        status: "answered",
        abstain_reason: null,
        timings: { total: 23.59 },
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockAnswer,
      });

      const res = await generateAnswer({
        question: "Do LEMS and GFWSVD describe factor estimation in the same way?",
        doc_ids: ["doc_c67573ef50cc", "doc_7fdb1200e823"],
        top_k: 8,
      });

      expect(res.status).toBe("answered");
      expect(res.comparisons).toHaveLength(1);
      expect(res.comparisons[0].comparison_type).toBe("unresolved");
      expect(res.evidence_links[0].quote).toContain("per-sample gradients");
    });

    it("uploadDocument passes FormData without setting Content-Type header", async () => {
      const mockUploadResponse: DocumentUploadResponse = {
        document: {
          doc_id: "doc_1234567890ab",
          title: "New Paper",
          filename: "new.pdf",
          sha256: "a".repeat(64),
          page_count: 10,
          extraction_warnings: [],
        },
        chunk_count: 25,
      };

      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => mockUploadResponse,
      });

      const file = new File([new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d])], "new.pdf", {
        type: "application/pdf",
      });

      const res = await uploadDocument(file);
      expect(res.chunk_count).toBe(25);

      const fetchCall = vi.mocked(global.fetch).mock.calls[0];
      const requestInit = fetchCall[1] as RequestInit;
      expect(requestInit.body).toBeInstanceOf(FormData);
      expect(requestInit.headers).toEqual({}); // Content-Type must NOT be set manually
    });
  });

  describe("Error Handling", () => {
    it("handles HTTP 400 Bad Request with string detail", async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: "Invalid doc_id format: 'bad_id'" }),
      });

      await expect(getDocument("bad_id")).rejects.toThrowError(ApiClientError);
      await expect(getDocument("bad_id")).rejects.toMatchObject({
        status: 400,
        detail: "Invalid doc_id format: 'bad_id'",
      });
    });

    it("handles HTTP 422 Unprocessable Entity with validation details array", async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({
          detail: [
            { loc: ["body", "question"], msg: "Field required", type: "missing" },
          ],
        }),
      });

      await expect(generateAnswer({ question: "" })).rejects.toMatchObject({
        status: 422,
        detail: "Field required",
      });
    });

    it("handles network failure gracefully", async () => {
      global.fetch = vi.fn().mockRejectedValueOnce(new Error("Failed to fetch"));

      await expect(getHealth()).rejects.toMatchObject({
        status: 0,
        detail: "Network error: Failed to fetch",
      });
    });

    it("handles invalid JSON response", async () => {
      global.fetch = vi.fn().mockResolvedValueOnce({
        ok: true,
        json: async () => {
          throw new Error("SyntaxError: Unexpected token");
        },
      });

      await expect(getHealth()).rejects.toMatchObject({
        detail: "Server returned invalid JSON response.",
      });
    });

    it("maps AbortError to 408 with exact timeout duration for generateAnswer (180s)", async () => {
      const abortError = new Error("The operation was aborted");
      abortError.name = "AbortError";
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);

      await expect(
        generateAnswer({ question: "Compare factor estimation" })
      ).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 180s",
      });
    });

    it("preserves established timeout durations for non-answer endpoints on abort", async () => {
      const abortError = new Error("The operation was aborted");
      abortError.name = "AbortError";

      // getHealth -> 5s
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);
      await expect(getHealth()).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 5s",
      });

      // listDocuments -> 10s
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);
      await expect(listDocuments()).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 10s",
      });

      // getDocument -> 10s
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);
      await expect(getDocument("doc_123")).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 10s",
      });

      // getDocumentChunks -> 15s
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);
      await expect(getDocumentChunks("doc_123")).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 15s",
      });

      // searchCorpus -> 15s
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);
      await expect(searchCorpus({ query: "test" })).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 15s",
      });

      // uploadDocument -> 90s
      global.fetch = vi.fn().mockRejectedValueOnce(abortError);
      const dummyFile = new File(["dummy"], "paper.pdf", { type: "application/pdf" });
      await expect(uploadDocument(dummyFile)).rejects.toMatchObject({
        status: 408,
        detail: "Request timed out after 90s",
      });
    });
  });
});
