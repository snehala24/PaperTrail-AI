from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

CHUNKER_VERSION: str = "0.2.1"


class ChunkKind(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE_REF = "figure_ref"
    REFERENCES = "references"


class Document(BaseModel):
    """Data contract representing an ingested document."""
    doc_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title or inferred heading")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    sha256: str = Field(..., description="SHA-256 hash of the original document content")
    page_count: int = Field(..., ge=0, description="Total number of pages in the document")
    chunker_version: Optional[str] = Field(
        default=None,
        description="Version of chunker algorithm used; permits re-ingestion on chunker updates"
    )
    extraction_warnings: List[str] = Field(
        default_factory=list,
        description="Recorded extraction warnings and parser issues; never silently dropped"
    )

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if len(v) != 64 or not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("sha256 must be a 64-character hexadecimal string")
        return v.lower()


class Chunk(BaseModel):
    """Data contract representing a granular chunk extracted from a document."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="Foreign key referencing the parent Document")
    page: int = Field(..., ge=1, description="1-indexed page number where this chunk appears")
    section_heading: Optional[str] = Field(
        default=None,
        description="Hierarchical section heading if identifiable"
    )
    text: str = Field(..., min_length=1, description="Extracted raw or cleaned text content")
    kind: ChunkKind = Field(
        default=ChunkKind.TEXT,
        description="Chunk type: text, table, figure_ref, or references"
    )
