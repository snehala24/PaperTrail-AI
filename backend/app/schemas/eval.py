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
    VERIFIED = "verified"


class BenchmarkQuestion(BaseModel):
    """Data contract for a benchmark evaluation question."""
    question_id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., min_length=5, description="Benchmark query text")
    question_type: BenchmarkQuestionType = Field(..., description="Query category")
    status: BenchmarkStatus = Field(default=BenchmarkStatus.DRAFT, description="Annotation status")
    target_doc_ids: List[str] = Field(
        default_factory=list,
        description="List of expected document IDs containing answers or evidence"
    )
    notes: Optional[str] = Field(default="", description="Annotator notes or instructions")


class GoldEvidence(BaseModel):
    """Data contract for ground-truth evidence linking."""
    gold_id: str = Field(..., description="Unique gold annotation identifier")
    question_id: str = Field(..., description="Target question identifier")
    chunk_id: str = Field(..., description="Target chunk identifier")
    expected_relation: EvidenceRelation = Field(..., description="Expected relation type")
    status: BenchmarkStatus = Field(default=BenchmarkStatus.DRAFT, description="Annotation status")
    conflict_reason: Optional[str] = Field(
        default=None,
        description="Reason for apparent conflict if relation is contradicts or weakens"
    )
    annotator_notes: Optional[str] = Field(default="", description="Annotation rationale")
