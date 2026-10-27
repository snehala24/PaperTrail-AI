from typing import Optional
from pydantic import BaseModel, Field, model_validator
from .claim import ClaimVerificationStatus


class CitationCheck(BaseModel):
    """
    Data contract for layered citation verification:
    Layer 1: Check existence in retrieved evidence set & document filter adherence
    Layer 2: Normalized verbatim quote match against chunk text
    Layer 3: Separate temperature-0 LLM entailment check
    """
    claim_id: str = Field(..., description="Identifier of the claim being cited")
    chunk_id: str = Field(..., description="Identifier of the cited chunk")
    exists: bool = Field(
        ...,
        description="Layer 1: Whether chunk exists in the retrieved context set and satisfies doc filter"
    )
    supports: Optional[bool] = Field(
        default=None,
        description="Layer 2 & 3: Whether chunk semantically supports the claim (None if not evaluated or Layer 1 fails)"
    )
    notes: str = Field(
        default="",
        description="Diagnostic or auditor notes regarding existence or support"
    )
    status: ClaimVerificationStatus = Field(
        default=ClaimVerificationStatus.UNVERIFIED,
        description="Detailed verification outcome status (verified, unsupported, invalid_reference, unverified)"
    )

    @model_validator(mode="after")
    def validate_support_requires_existence(self) -> "CitationCheck":
        # Layer 1 failure invariant: When absent from retrieved evidence set or violating document filter,
        # supports MUST be None and status MUST be invalid_reference.
        if not self.exists:
            if self.supports is not None:
                raise ValueError(
                    "Citation support must be None if the chunk does not exist in the retrieved evidence set (exists=False)."
                )
            self.status = ClaimVerificationStatus.INVALID_REFERENCE
        else:
            # When exists is True, citation is a valid reference so status cannot be invalid_reference
            if self.status == ClaimVerificationStatus.INVALID_REFERENCE:
                raise ValueError("Citation cannot have status 'invalid_reference' when exists=True.")

            # Final verification invariant: status cannot be 'verified' unless supports is True
            # (supports=True alone does NOT auto-promote to verified; verifier must confirm all layers passed)
            if self.status == ClaimVerificationStatus.VERIFIED and self.supports is not True:
                raise ValueError("Citation cannot have status 'verified' unless exists=True and supports=True.")

            # Inconsistency guard: cannot mark unsupported if supports is True
            if self.status == ClaimVerificationStatus.UNSUPPORTED and self.supports is True:
                raise ValueError("Citation cannot have status 'unsupported' when supports=True.")

        return self
