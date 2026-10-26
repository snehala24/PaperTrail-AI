import pytest
from backend.app.schemas import Chunk, ChunkKind
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.hybrid import HybridRetriever


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="chk-1",
            doc_id="doc-1",
            page=1,
            section_heading="1. Attention",
            text="The Transformer relies entirely on multi-head self-attention mechanisms without recurrence.",
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="chk-2",
            doc_id="doc-1",
            page=2,
            section_heading="2. Results",
            text="On the WMT 2014 English-to-German task, the base model achieved 28.4 BLEU.",
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="chk-3",
            doc_id="doc-2",
            page=1,
            section_heading="References",
            text="[1] Vaswani et al. Attention is all you need. In NeurIPS 2017.",
            kind=ChunkKind.REFERENCES,
        ),
    ]


def test_bm25_index_search(sample_chunks):
    bm25 = BM25Index()
    bm25.build(sample_chunks)

    results = bm25.search("BLEU German", top_k=2)
    assert len(results) >= 1
    assert results[0][0].chunk_id == "chk-2"
    assert results[0][1] > 0.0  # Positive BM25 score


def test_reciprocal_rank_fusion():
    """Verify RRF formula: score = sum(1.0 / (k + rank))."""
    # Item 1 is rank 1 in list A, rank 2 in list B
    # Item 2 is rank 2 in list A, rank 1 in list B
    # Item 3 is rank 3 in list A, absent in list B
    ranks_bm25 = [("chk-1", 1), ("chk-2", 2), ("chk-3", 3)]
    ranks_dense = [("chk-2", 1), ("chk-1", 2)]

    fused = reciprocal_rank_fusion(
        rankings=[ranks_bm25, ranks_dense],
        k=60,
    )

    # chk-1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # chk-2: 1/(60+2) + 1/(60+1) = 0.03252 (tied)
    # chk-3: 1/(60+3) = 0.01587
    fused_dict = dict(fused)
    assert fused_dict["chk-1"] > fused_dict["chk-3"]
    assert fused_dict["chk-2"] > fused_dict["chk-3"]


def test_hybrid_retriever_excludes_references_by_default(sample_chunks):
    retriever = HybridRetriever()
    retriever.index_chunks(sample_chunks)

    # Search for Vaswani, which is mentioned in references chunk chk-3
    results = retriever.search("Vaswani", top_k=5, include_references=False)
    # References should be excluded
    returned_ids = [r.chunk_id for r in results]
    assert "chk-3" not in returned_ids

    # With include_references=True, references can be returned
    results_with_refs = retriever.search("Vaswani", top_k=5, include_references=True)
    returned_ids_with_refs = [r.chunk_id for r in results_with_refs]
    assert "chk-3" in returned_ids_with_refs


def test_unique_exact_match_survives_into_rerank_pool():
    """
    Verify that a chunk with a unique exact-match token (BM25 rank 1, dense rank low/absent)
    survives into the fused candidate pool (size RERANK_POOL=30) passed to the cross-encoder.
    """
    # 40 distractors containing generic optimization prose
    distractors = [
        Chunk(
            chunk_id=f"distractor-{i}",
            doc_id="doc-synthetic",
            page=i + 1,
            section_heading="General Optimization",
            text=f"Deep neural networks require parameter optimization, gradient descent, and learning rate schedules step {i}.",
            kind=ChunkKind.TEXT,
        )
        for i in range(40)
    ]

    # Target chunk with a unique exact-match keyword not present in any distractor
    target = Chunk(
        chunk_id="target-unique-solver",
        doc_id="doc-synthetic",
        page=41,
        section_heading="Specific Binary",
        text="Internal tooling: GurobiSolverXYZ.",
        kind=ChunkKind.TEXT,
    )
    corpus = distractors + [target]

    bm25 = BM25Index()
    bm25.build(corpus)

    dense = DenseIndex()
    dense.build(corpus)

    query = "Deep neural network training gradient descent learning rate scheduling GurobiSolverXYZ"

    # 1. BM25 rank 1 for exact token
    bm25_raw = bm25.search(query, top_k=30)
    assert bm25_raw[0][0].chunk_id == "target-unique-solver"
    assert bm25_raw[0][2] == 1  # Rank 1

    # 2. Dense rank is low or absent from top 10
    dense_raw = dense.search(query, top_k=30)
    dense_target_rank = next((r for c, _, r in dense_raw if c.chunk_id == "target-unique-solver"), 999)
    assert dense_target_rank > 10

    # 3. Reciprocal Rank Fusion of top 30 from both
    bm25_ranks = [(c.chunk_id, r) for c, _, r in bm25_raw[:30]]
    dense_ranks = [(c.chunk_id, r) for c, _, r in dense_raw[:30]]
    fused = reciprocal_rank_fusion([bm25_ranks, dense_ranks], k=60)
    fused_pool_cids = [cid for cid, _ in fused][:30]

    # 4. Assert target chunk survives into the fused candidate pool
    assert "target-unique-solver" in fused_pool_cids
