import pytest
from unittest.mock import MagicMock, patch
from backend.app.schemas import Chunk, ChunkKind, ScoredChunk, SearchRequest, SearchResponse
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.api.search import search_corpus


def test_bm25_search_empty_corpus_returns_empty():
    """B1: BM25Index.search on empty corpus returns [] and does not raise ZeroDivisionError."""
    idx = BM25Index()
    idx.build([])
    assert idx.search("transformer attention") == []

    # Without build() called at all
    idx_unbuilt = BM25Index()
    assert idx_unbuilt.search("any query") == []


def test_bm25_build_empty_tokens_does_not_raise_zero_division():
    """B1: BM25Index.build on chunks with zero alphanumeric tokens does not raise ZeroDivisionError."""
    idx = BM25Index()
    empty_chunks = [
        Chunk(chunk_id="c1", doc_id="d1", page=1, text="... --- !!!", kind=ChunkKind.TEXT),
        Chunk(chunk_id="c2", doc_id="d1", page=1, text="   \n\t  ", kind=ChunkKind.TEXT),
    ]
    idx.build(empty_chunks)
    assert idx.search("transformer") == []


def test_bm25_search_zero_matching_tokens_returns_empty():
    """B1: BM25Index.search on query with zero matching tokens returns [] without raising."""
    idx = BM25Index()
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", page=1, text="The transformer architecture uses attention.", kind=ChunkKind.TEXT),
    ]
    idx.build(chunks)
    # Query completely disjoint from corpus tokens
    results = idx.search("quantum teleportation superconductor")
    assert results == []


def test_rrf_math_exact_scores():
    """B2: RRF formula exact match test: rank 1 has score 1/(60+1) ≈ 0.016393; multi-list ranks match pytest.approx."""
    # Single list, rank 1
    single = reciprocal_rank_fusion([[("docA", 1)]], k=60)
    assert len(single) == 1
    assert single[0][0] == "docA"
    assert single[0][1] == pytest.approx(1.0 / 61.0, rel=1e-5)
    assert single[0][1] == pytest.approx(0.01639344, rel=1e-5)

    # Two lists with known ranks
    ranks_bm25 = [("c1", 1), ("c2", 2), ("c3", 3)]
    ranks_dense = [("c2", 1), ("c1", 2)]

    fused = reciprocal_rank_fusion([ranks_bm25, ranks_dense], k=60)
    fused_dict = dict(fused)

    # c1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.032522
    expected_c1 = (1.0 / 61.0) + (1.0 / 62.0)
    assert fused_dict["c1"] == pytest.approx(expected_c1, rel=1e-5)
    assert fused_dict["c1"] == pytest.approx(0.03252247, rel=1e-5)

    # c2: 1/(60+2) + 1/(60+1)
    expected_c2 = (1.0 / 62.0) + (1.0 / 61.0)
    assert fused_dict["c2"] == pytest.approx(expected_c2, rel=1e-5)

    # c3: 1/(60+3) = 1/63 ≈ 0.015873
    expected_c3 = 1.0 / 63.0
    assert fused_dict["c3"] == pytest.approx(expected_c3, rel=1e-5)
    assert fused_dict["c3"] == pytest.approx(0.01587302, rel=1e-5)


def test_hybrid_search_bm25_empty_dense_still_produces_results():
    """B1: If BM25 returns [], dense search still runs and produces results."""
    retriever = HybridRetriever()
    chunk_dense = Chunk(
        chunk_id="chunk_dense_only",
        doc_id="doc_1",
        page=1,
        text="Dense embedding semantic match content.",
        kind=ChunkKind.TEXT,
    )
    retriever.chunks_by_id = {"chunk_dense_only": chunk_dense}

    # Mock BM25 returning empty list
    retriever.bm25.search = MagicMock(return_value=[])
    # Mock Dense returning candidate
    retriever.dense.search = MagicMock(return_value=[(chunk_dense, 0.85, 1)])
    # Mock Reranker
    retriever.reranker.rerank = MagicMock(return_value=[(chunk_dense, 0.95, 1)])

    results = retriever.search("unmatched query", top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_dense_only"
    assert results[0].bm25_rank is None
    assert results[0].dense_rank == 1
    assert results[0].fused_rank == 1
    assert retriever.last_total_candidates == 1


def test_search_response_contract_counts():
    """B3: SearchResponse schema includes total_candidates and returned_count."""
    scored_chunk = ScoredChunk(
        chunk_id="chk_1",
        doc_id="doc_1",
        page=1,
        text="Sample retrieved text.",
        kind=ChunkKind.TEXT,
    )

    resp = SearchResponse(
        query="test query",
        results=[scored_chunk],
        total_candidates=15,
        returned_count=1,
    )
    assert resp.total_candidates == 15
    assert resp.returned_count == 1
    assert resp.returned_count == len(resp.results)


def test_search_api_endpoint_populates_returned_count():
    """B3: POST /search populates total_candidates and returned_count."""
    sample_chunk = ScoredChunk(
        chunk_id="chk_test",
        doc_id="doc_0123456789ab",
        page=1,
        text="Evaluated chunk text.",
        kind=ChunkKind.TEXT,
    )

    with patch("backend.app.api.search.retriever.search", return_value=[sample_chunk]):
        with patch("backend.app.api.search.retriever.last_total_candidates", 25):
            req = SearchRequest(query="sample search", top_k=1)
            response = search_corpus(req)

            assert response.total_candidates == 25
            assert response.returned_count == 1
            assert len(response.results) == 1
