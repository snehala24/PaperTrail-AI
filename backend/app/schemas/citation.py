from typing import Optional
from pydantic import BaseModel, Field, model_validator


class CitationCheck(BaseModel):
    """
    Data contract for the two-step citation validator:
    Step 1: Check existence (does the chunk_id exist in the corpus / retrieved context?)
    Step 2: Check support (does the chunk actually support the claim?)
    """
    claim_id: str = Field(..., description="Identifier of the claim being cited")
    chunk_id: str = Field(..., description="Identifier of the cited chunk")
    exists: bool = Field(..., description="Step 1: Whether the referenced chunk_id exists")
    supports: Optional[bool] = Field(
        default=None,
        description="Step 2: Whether the chunk semantically supports the claim (None if not yet evaluated or doesn't exist)"
    )
    notes: str = Field(
        default="",
        description="Diagnostic or auditor notes regarding existence or support"
    )

    @model_validator(mode="after")
    def validate_support_requires_existence(self) -> "CitationCheck":
        # If the chunk does not exist, it cannot support the claim
        if not self.exists and self.supports is True:
            raise ValueError("Citation cannot support a claim if the chunk does not exist in the corpus (exists=False).")
        return self
