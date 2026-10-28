from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    WEAKENS = "weakens"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    INSUFFICIENT = "insufficient"


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
    status: ClaimVerificationStatus = Field(
        default=ClaimVerificationStatus.UNVERIFIED,
        description="Verification status outcome for this claim"
    )


class EvidenceLink(BaseModel):
    """
    Data contract linking an atomic claim to a specific document chunk.
    Captures the relationship, rationalized confidence, and conflict reasons.
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
