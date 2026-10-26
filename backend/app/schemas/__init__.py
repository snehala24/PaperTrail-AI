"""Pydantic v2 data contracts for PaperTrail AI."""
from .document import Document, Chunk, ChunkKind
from .claim import Claim, EvidenceLink, EvidenceRelation
from .citation import CitationCheck
from .answer import Answer
from .api import HealthResponse, DocumentUploadResponse

__all__ = [
    "Document",
    "Chunk",
    "ChunkKind",
    "Claim",
    "EvidenceLink",
    "EvidenceRelation",
    "CitationCheck",
    "Answer",
    "HealthResponse",
    "DocumentUploadResponse",
]
