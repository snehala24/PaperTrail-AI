import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.claim import EvidenceRelation


class BenchmarkQuestionType(str, Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    APPARENT_CONFLICT = "apparent_conflict"
    UNANSWERABLE = "unanswerable"
    METHODOLOGICAL = "methodological"


class BenchmarkStatus(str, Enum):
    DRAFT = "draft"
    AI_DRAFT = "ai_draft"
    VERIFIED = "verified"
    HUMAN_VERIFIED = "human_verified"


class BenchmarkQuestion(BaseModel):
    """Data contract for a benchmark evaluation question."""
    question_id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., min_length=5, description="Benchmark query text")
    question_type: BenchmarkQuestionType = Field(..., description="Query category")
    status: BenchmarkStatus = Field(default=BenchmarkStatus.AI_DRAFT, description="Annotation status")
    labelled_by: Optional[str] = Field(default="assistant", description="Provenance of annotation (e.g. assistant, human)")
    target_doc_ids: List[str] = Field(
        default_factory=list,
        description="List of expected document IDs containing answers or evidence"
    )
    gold_chunk_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of explicit gold chunk IDs spanning one or multiple documents"
    )
    notes: Optional[str] = Field(default="", description="Annotator notes or instructions")


class GoldEvidence(BaseModel):
    """
    Data contract for ground-truth evidence linking.
    Uses (doc_sha256, page, answer_span) rather than dynamic chunk_ids so that
    benchmark annotations survive chunker modifications.
    """
    gold_id: str = Field(..., description="Unique gold annotation identifier")
    question_id: str = Field(..., description="Target question identifier")
    doc_sha256: str = Field(..., description="SHA-256 hash of the target document")
    page: int = Field(..., ge=1, description="1-indexed page number containing the evidence")
    answer_span: str = Field(..., min_length=1, description="Exact or normalized text span of the evidence")
    expected_relation: EvidenceRelation = Field(default=EvidenceRelation.SUPPORTS, description="Expected relation type")
    status: BenchmarkStatus = Field(default=BenchmarkStatus.AI_DRAFT, description="Annotation status")
    labelled_by: Optional[str] = Field(default="assistant", description="Provenance of annotation")
    gold_chunk_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of explicit gold chunk IDs for multi-doc evaluation"
    )
    conflict_reason: Optional[str] = Field(
        default=None,
        description="Reason for apparent conflict if relation is contradicts or weakens"
    )
    annotator_notes: Optional[str] = Field(default="", description="Annotation rationale")


def normalize_text_span(text: str) -> str:
    """
    Normalize hyphens and whitespace on both sides for robust span matching:
    1. Replaces unicode dashes/hyphens with ASCII '-'
    2. Joins line-break hyphens ("pow-\nerful" or "pow- erful" -> "powerful") when next char is lowercase
    3. Normalizes whitespace and lowercases
    """
    if not text:
        return ""
    # Replace various unicode hyphens/dashes with standard ASCII '-'
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    # Join line-break hyphens ("pow-\nerful" -> "powerful")
    t = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-z][a-zA-Z]*)", r"\1\2", t)
    # Also handle space-separated line-break hyphen remnants ("pow- erful" -> "powerful")
    t = re.sub(r"(\b[a-zA-Z]+)-\s+([a-z][a-zA-Z]*)", r"\1\2", t)
    # Normalize multiple whitespace characters to single space and lowercase
    return re.sub(r"\s+", " ", t).strip().lower()


def is_chunk_match(chunk_page: int, chunk_text: str, gold_page: int, answer_span: str) -> bool:
    """
    Determine if a chunk matches gold evidence:
    Chunk must be on the same page and contain the answer span (normalized hyphens and whitespace, case-insensitive).
    """
    if chunk_page != gold_page:
        return False
    norm_chunk = normalize_text_span(chunk_text)
    norm_span = normalize_text_span(answer_span)
    return norm_span in norm_chunk
