import re
from fastapi import APIRouter, HTTPException, status
from backend.app.schemas import SearchRequest, SearchResponse
from backend.app.retrieval.hybrid import retriever

router = APIRouter(prefix="/search", tags=["Search"])

DOC_ID_REGEX = re.compile(r"^doc_[0-9a-f]{12}$")


@router.post("", response_model=SearchResponse)
def search_corpus(req: SearchRequest) -> SearchResponse:
    """
    Execute hybrid retrieval (BM25 + Dense FAISS via RRF) and Cross-Encoder reranking.
    Excludes references by default unless include_references=True.
    Returns per-stage scores and ranks for transparency.
    """
    if req.doc_ids:
        for did in req.doc_ids:
            if not DOC_ID_REGEX.match(did):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid doc_id in filter: {did}. Expected format '^doc_[0-9a-f]{{12}}$'.",
                )

    results = retriever.search(
        query=req.query,
        top_k=req.top_k,
        doc_ids=req.doc_ids,
        include_references=req.include_references,
        use_reranker=True,
    )
    total_candidates = getattr(retriever, "last_total_candidates", len(results))

    return SearchResponse(
        query=req.query,
        results=results,
        total_candidates=total_candidates,
        returned_count=len(results),
    )
