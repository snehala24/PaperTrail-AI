/**
 * TypeScript Data Contracts for PaperTrail AI
 * 
 * Faithfully mirrors backend Pydantic models from:
 * - backend/app/schemas/document.py
 * - backend/app/schemas/claim.py
 * - backend/app/schemas/citation.py
 * - backend/app/schemas/answer.py
 * - backend/app/schemas/api.py
 * - backend/app/schemas/retrieval.py
 */

// ==========================================
// Document & Ingestion Schemas
// ==========================================

export type ChunkKind =
  | "text"
  | "table"
  | "figure_ref"
  | "figure_text"
  | "footnote"
  | "references";

export interface Document {
  /** Unique identifier for the document, e.g. doc_c67573ef50cc */
  doc_id: string;
  /** Inferred or metadata title */
  title: string;
  /** Original filename of the uploaded PDF */
  filename: string;
  /** 64-character SHA-256 hash */
  sha256: string;
  /** Total number of pages extracted */
  page_count: number;
  /** Chunker version tag */
  chunker_version?: string | null;
  /** Recorded extraction warnings and issues */
  extraction_warnings: string[];
}

export interface Chunk {
  /** Unique chunk identifier, e.g. doc_c67573ef50cc_p3_c5 */
  chunk_id: string;
  /** Document ID referencing the parent Document */
  doc_id: string;
  /** 1-indexed page number */
  page: number;
  /** Section heading if identified */
  section_heading?: string | null;
  /** Extracted chunk text */
  text: string;
  /** Chunk semantic kind */
  kind: ChunkKind;
}

export interface DocumentUploadResponse {
  /** Ingested document metadata and diagnostics */
  document: Document;
  /** Total number of chunks extracted and indexed */
  chunk_count: number;
}

export interface HealthResponse {
  /** Service health status ('ok') */
  status: string;
  /** Backend application version */
  version: string;
  /** ISO UTC timestamp */
  timestamp: string;
}

// ==========================================
// Claims, Citations, & Verification Schemas
// ==========================================

export type AnswerStatus = "answered" | "abstained" | "error";

export type EvidenceRelation =
  | "supports"
  | "weakens"
  | "contradicts"
  | "contextualizes"
  | "insufficient";

export type ComparisonType =
  | "supports"
  | "contradicts"
  | "apparent_conflict"
  | "unresolved";

export type ClaimVerificationStatus =
  | "verified"
  | "unsupported"
  | "invalid_reference"
  | "unverified";

export interface EvidenceLink {
  /** Claim identifier being evaluated */
  claim_id: string;
  /** Identifier of the supporting/contradicting chunk */
  chunk_id: string;
  /** Nature of evidence relationship */
  relation: EvidenceRelation;
  /** Explanation for the relationship */
  rationale: string;
  /** Calibrated confidence score (0.0 to 1.0) */
  confidence: number;
  /** Reason for apparent conflict (differing experimental conditions, datasets, metrics) */
  conflict_reason?: string | null;
  /** Identifier of the source document */
  doc_id?: string | null;
  /** 1-indexed page number */
  page?: number | null;
  /** Short verbatim quote extracted from cited chunk */
  quote?: string | null;
}

export interface CitationCheck {
  /** Identifier of the claim being cited */
  claim_id: string;
  /** Identifier of the cited chunk */
  chunk_id: string;
  /** Layer 1: Exists in retrieved context set and satisfies doc filter */
  exists: boolean;
  /** Layer 2 & 3: Semantically supports the claim (null if Layer 1 fails or unevaluated) */
  supports?: boolean | null;
  /** Diagnostic or audit notes */
  notes: string;
  /** Verification status */
  status: ClaimVerificationStatus;
}

export interface ClaimComparison {
  /** Identifier of first claim or cited chunk */
  claim_id_a: string;
  /** Identifier of second claim or cited chunk */
  claim_id_b: string;
  /** Source document ID of first claim */
  doc_id_a?: string | null;
  /** Source document ID of second claim */
  doc_id_b?: string | null;
  /** Comparison classification */
  comparison_type: ComparisonType;
  /** Summary explanation of the cross-paper comparison */
  summary: string;
  /** Reason for apparent conflict */
  conflict_reason?: string | null;
}

// ==========================================
// Answer Service Schemas
// ==========================================

export interface AnswerRequest {
  /** Natural language research query */
  question: string;
  /** Optional filter restricted to specific document IDs (format: ^doc_[0-9a-f]{12}$) */
  doc_ids?: string[] | null;
  /** Evidence context depth (1 to 10, default 8) */
  top_k?: number;
}

export interface AnswerResponse {
  /** Synthesized answer text or honest abstention explanation */
  answer: string;
  /** Evidence links connecting claims to supporting chunks */
  evidence_links: EvidenceLink[];
  /** 3-layer citation verification checks */
  citation_checks: CitationCheck[];
  /** Structured cross-paper comparisons */
  comparisons: ClaimComparison[];
  /** Response status: 'answered', 'abstained', or 'error' */
  status: AnswerStatus;
  /** Reason for abstention when status === 'abstained' */
  abstain_reason?: string | null;
  /** Pipeline latency breakdown in seconds */
  timings: Record<string, number>;
}

// ==========================================
// Retrieval & Search Schemas
// ==========================================

export interface SearchRequest {
  /** Natural language query */
  query: string;
  /** Number of results to return (1 to 100, default 10) */
  top_k?: number;
  /** Optional document ID filters */
  doc_ids?: string[] | null;
  /** Whether to include bibliography/references chunks */
  include_references?: boolean;
}

export interface ScoredChunk {
  chunk_id: string;
  doc_id: string;
  page: number;
  section_heading?: string | null;
  text: string;
  kind: ChunkKind;
  bm25_score?: number | null;
  bm25_rank?: number | null;
  dense_score?: number | null;
  dense_rank?: number | null;
  fused_score?: number | null;
  fused_rank?: number | null;
  rerank_score?: number | null;
  rerank_rank?: number | null;
}

export interface SearchResponse {
  query: string;
  results: ScoredChunk[];
  total_candidates: number;
  returned_count: number;
}

// ==========================================
// API Error Schema
// ==========================================

export interface ApiErrorDetail {
  loc?: (string | number)[];
  msg: string;
  type?: string;
}

export interface ApiErrorResponse {
  detail: string | ApiErrorDetail[];
}
