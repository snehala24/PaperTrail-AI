import re
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from .claim import Claim, EvidenceLink
from .citation import CitationCheck

DOC_ID_REGEX = re.compile(r"^doc_[0-9a-f]{12}$")


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ERROR = "error"


class AnswerRequest(BaseModel):
    """
    Request specification for grounded research question answering.
    Supports natural language queries, document filtering, and bounded retrieval depth.
    """
    question: str = Field(..., min_length=1, description="Natural language research query")
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional filter restricted to specific document IDs (format: ^doc_[0-9a-f]{12}$)"
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=10,
        description="Bounded evidence context depth to retrieve and present (maximum 10)"
    )

    @field_validator("question")
    @classmethod
    def validate_question_non_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("question must not be empty or whitespace only")
        return cleaned

    @field_validator("doc_ids")
    @classmethod
    def validate_doc_ids_format(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for did in v:
                if not DOC_ID_REGEX.match(did):
                    raise ValueError(
                        f"Invalid doc_id in filter: '{did}'. Expected format '^doc_[0-9a-f]{{12}}$'."
                    )
        return v


class AnswerResponse(BaseModel):
    """
    Data contract representing a grounded answer response with complete verification audit trail.
    """
    answer: str = Field(..., description="Synthesized answer text or honest abstention explanation")
    evidence_links: List[EvidenceLink] = Field(
        default_factory=list,
        description="Evidence links between claims and chunks"
    )
    citation_checks: List[CitationCheck] = Field(
        default_factory=list,
        description="Audit checks verifying existence, quotation, and semantic support for citations"
    )
    status: AnswerStatus = Field(..., description="High-level response status: answered, abstained, or error")
    abstain_reason: Optional[str] = Field(
        default=None,
        description="Explicit reason for abstention (required when status='abstained')"
    )
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution latencies for pipeline stages in seconds"
    )

    @model_validator(mode="after")
    def validate_abstention_consistency(self) -> "AnswerResponse":
        if self.status == AnswerStatus.ABSTAINED:
            if not self.abstain_reason or not self.abstain_reason.strip():
                raise ValueError("abstain_reason is required when status is 'abstained'")
        return self


class Answer(BaseModel):
    """
    Data contract representing a synthesized response to a research question.
    Contains atomic claims and honest abstention metadata.
    """
    answer_id: str = Field(..., description="Unique identifier for the generated answer")
    question: str = Field(..., min_length=1, description="Original research query")
    claims: List[Claim] = Field(
        default_factory=list,
        description="List of atomic claims parsed from the generated answer"
    )
    abstained: bool = Field(
        default=False,
        description="Whether the system abstained due to insufficient or conflicting evidence"
    )
    abstain_reason: Optional[str] = Field(
        default=None,
        description="Explicit reason for abstention (required when abstained=True)"
    )

    @model_validator(mode="after")
    def validate_abstention(self) -> "Answer":
        if self.abstained and (not self.abstain_reason or not self.abstain_reason.strip()):
            raise ValueError("abstain_reason is required when abstained=True")
        return self
