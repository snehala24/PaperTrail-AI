from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from .claim import Claim


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
