/**
 * Centralized Typed API Client for PaperTrail AI
 * 
 * Communicates with the FastAPI backend.
 * Endpoints:
 * - GET  /health
 * - POST /documents
 * - GET  /documents
 * - GET  /documents/{doc_id}
 * - GET  /documents/{doc_id}/chunks
 * - POST /search
 * - POST /answer
 */

import {
  HealthResponse,
  Document,
  Chunk,
  DocumentUploadResponse,
  AnswerRequest,
  AnswerResponse,
  SearchRequest,
  SearchResponse,
  ApiErrorDetail,
} from "./types";

export class ApiClientError extends Error {
  public readonly status: number;
  public readonly detail: string;
  public readonly errorDetails?: ApiErrorDetail[];

  constructor(status: number, detail: string, errorDetails?: ApiErrorDetail[]) {
    super(`API Error ${status}: ${detail}`);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
    this.errorDetails = errorDetails;
  }
}

export function getApiBaseUrl(): string {
  if (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/+$/, "");
  }
  return "http://localhost:8000";
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

async function request<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

  const { timeoutMs = 60000, headers = {}, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle non-2xx responses
    if (!res.ok) {
      let detail = `Request failed with status ${res.status}`;
      let errorDetails: ApiErrorDetail[] | undefined;

      try {
        const errorBody = await res.json();
        if (typeof errorBody.detail === "string") {
          detail = errorBody.detail;
        } else if (Array.isArray(errorBody.detail)) {
          errorDetails = errorBody.detail;
          detail = errorBody.detail.map((e: ApiErrorDetail) => e.msg).join("; ");
        }
      } catch {
        // Body was not JSON
        detail = res.statusText || detail;
      }

      throw new ApiClientError(res.status, detail, errorDetails);
    }

    // Parse JSON
    try {
      return (await res.json()) as T;
    } catch {
      throw new ApiClientError(res.status, "Server returned invalid JSON response.");
    }
  } catch (err: unknown) {
    clearTimeout(timeoutId);

    if (err instanceof ApiClientError) {
      throw err;
    }

    if (err instanceof Error) {
      if (err.name === "AbortError") {
        throw new ApiClientError(408, `Request timed out after ${timeoutMs / 1000}s`);
      }
      throw new ApiClientError(0, `Network error: ${err.message}`);
    }

    throw new ApiClientError(0, "An unknown network error occurred.");
  }
}

// ==========================================
// API Methods
// ==========================================

/**
 * Health check endpoint.
 * Returns service status, version, and server timestamp.
 */
export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health", {
    method: "GET",
    timeoutMs: 5000,
  });
}

/**
 * Upload and ingest a research paper in PDF format.
 * Uses multipart/form-data. Do NOT set Content-Type header manually.
 */
export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return request<DocumentUploadResponse>("/documents", {
    method: "POST",
    body: formData,
    timeoutMs: 90000, // Ingestion can take up to 90s for dense PDFs
  });
}

/**
 * Retrieve list of all ingested research documents.
 */
export async function listDocuments(): Promise<Document[]> {
  return request<Document[]>("/documents", {
    method: "GET",
    timeoutMs: 10000,
  });
}

/**
 * Retrieve metadata for a specific document by its doc_id.
 */
export async function getDocument(docId: string): Promise<Document> {
  return request<Document>(`/documents/${encodeURIComponent(docId)}`, {
    method: "GET",
    timeoutMs: 10000,
  });
}

/**
 * Retrieve all granular chunks for a given document.
 */
export async function getDocumentChunks(docId: string): Promise<Chunk[]> {
  return request<Chunk[]>(`/documents/${encodeURIComponent(docId)}/chunks`, {
    method: "GET",
    timeoutMs: 15000,
  });
}

/**
 * Execute hybrid retrieval across the corpus.
 */
export async function searchCorpus(req: SearchRequest): Promise<SearchResponse> {
  return request<SearchResponse>("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(req),
    timeoutMs: 15000,
  });
}

/**
 * Generate an evidence-grounded answer to a scientific research question.
 * Runs hybrid retrieval, reranking, Gemini synthesis, 3-layer verification,
 * and zero-tolerance abstention.
 */
export async function generateAnswer(req: AnswerRequest): Promise<AnswerResponse> {
  return request<AnswerResponse>("/answer", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(req),
    timeoutMs: 180000, // LLM synthesis and 3-layer verification across multi-paper corpora
  });
}
