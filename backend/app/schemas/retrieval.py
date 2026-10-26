from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.document import ChunkKind


class SearchRequest(BaseModel):
    """Search request specification."""
    query: str = Field(..., min_length=1, description="Natural language search or research query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional filter to specific document IDs")
    include_references: bool = Field(
        default=False,
        description="Whether to include references/bibliography chunks (excluded by default)"
    )


class ScoredChunk(BaseModel):
    """A retrieved chunk enriched with multi-stage ranking metrics."""
    chunk_id: str
    doc_id: str
    page: int
    section_heading: Optional[str] = None
    text: str
    kind: ChunkKind
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    fused_score: Optional[float] = None
    fused_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None


class SearchResponse(BaseModel):
    """Search response returning scored and reranked chunks."""
    query: str
    results: List[ScoredChunk]
    total_candidates: int = Field(..., description="Size of pre-top_k candidate pool")
    returned_count: int = Field(..., description="Number of results actually returned (len(results))")
