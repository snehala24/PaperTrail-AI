from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    WEAKENS = "weakens"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    INSUFFICIENT = "insufficient"


class ComparisonType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    APPARENT_CONFLICT = "apparent_conflict"
    UNRESOLVED = "unresolved"


class ClaimVerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNSUPPORTED = "unsupported"
    INVALID_REFERENCE = "invalid_reference"
    UNVERIFIED = "unverified"


class Claim(BaseModel):
    """Data contract representing an atomic statement or claim extracted from an answer."""
    claim_id: str = Field(..., description="Unique identifier for the claim")
    answer_id: str = Field(..., description="Identifier of the answer containing this claim")
    text: str = Field(..., min_length=1, description="Atomic claim statement text")
    chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of cited context chunk IDs supporting this claim"
    )
    quote: Optional[str] = Field(
        default=None,
        description="Short verbatim evidence quote extracted from cited chunk"
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Identifier of primary source document supporting this claim"
    )
    page: Optional[int] = Field(
        default=None,
        description="1-indexed page number of primary cited chunk"
    )
    status: ClaimVerificationStatus = Field(
        default=ClaimVerificationStatus.UNVERIFIED,
        description="Verification status outcome for this claim"
    )


class EvidenceLink(BaseModel):
    """
    Data contract linking an atomic claim to a specific document chunk.
    Captures the relationship, rationalized confidence, conflict reasons, and source metadata.
    """
    claim_id: str = Field(..., description="Identifier of the claim being evaluated")
    chunk_id: str = Field(..., description="Identifier of the supporting/contradicting chunk")
    relation: EvidenceRelation = Field(..., description="Nature of the evidence relationship")
    rationale: str = Field(..., min_length=1, description="Explanation for the assigned relationship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score (0.0 to 1.0)")
    conflict_reason: Optional[str] = Field(
        default=None,
        description="Reason for apparent conflict (e.g. dataset, metric, experimental conditions)"
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Identifier of the source document containing this chunk"
    )
    page: Optional[int] = Field(
        default=None,
        description="1-indexed page number of the source chunk"
    )
    quote: Optional[str] = Field(
        default=None,
        description="Short verbatim evidence quote extracted from cited chunk"
    )

    @model_validator(mode="after")
    def validate_conflict_reason(self) -> "EvidenceLink":
        # Rule 3: Conflicts are labelled "apparent" with a reason; never assert definitive contradiction without rationale
        if self.relation in (EvidenceRelation.CONTRADICTS, EvidenceRelation.WEAKENS):
            if not self.conflict_reason or not self.conflict_reason.strip():
                raise ValueError(
                    f"conflict_reason is required when relation is '{self.relation.value}' "
                    "to document the apparent conflict cause (dataset, metric, conditions)."
                )
        return self


class ClaimComparison(BaseModel):
    """
    Data contract comparing claims or evidence across papers.
    Captures support, contradiction, or apparent conflict caused by differing experimental conditions.
    """
    claim_id_a: str = Field(..., description="Identifier of the first claim or cited chunk")
    claim_id_b: str = Field(..., description="Identifier of the second claim or cited chunk")
    doc_id_a: Optional[str] = Field(default=None, description="Document ID of first source")
    doc_id_b: Optional[str] = Field(default=None, description="Document ID of second source")
    comparison_type: ComparisonType = Field(..., description="Nature of comparison: supports, contradicts, apparent_conflict, unresolved")
    summary: str = Field(..., min_length=1, description="Summary explanation of the cross-paper comparison")
    conflict_reason: Optional[str] = Field(
        default=None,
        description="Reason for apparent conflict (e.g. differing experimental conditions, metrics, datasets)"
    )
