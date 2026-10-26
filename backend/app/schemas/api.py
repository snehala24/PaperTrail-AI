from datetime import datetime, timezone
from pydantic import BaseModel, Field
from .document import Document


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="ok", description="Health status of the API service")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Server UTC timestamp"
    )


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful ingestion of a research PDF."""
    document: Document = Field(..., description="Document metadata and extraction diagnostics")
    chunk_count: int = Field(..., ge=0, description="Total number of chunks extracted from the document")
