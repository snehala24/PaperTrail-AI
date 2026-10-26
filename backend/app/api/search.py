from fastapi import APIRouter
from backend.app.schemas import SearchRequest, SearchResponse
from backend.app.retrieval.hybrid import retriever

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search_corpus(req: SearchRequest) -> SearchResponse:
    """
    Execute hybrid retrieval (BM25 + Dense FAISS via RRF) and Cross-Encoder reranking.
    Excludes references by default unless include_references=True.
    Returns per-stage scores and ranks for transparency.
    """
    results = retriever.search(
        query=req.query,
        top_k=req.top_k,
        doc_ids=req.doc_ids,
        include_references=req.include_references,
        use_reranker=True,
    )

    return SearchResponse(
        query=req.query,
        results=results,
        total_candidates=len(results),
    )
