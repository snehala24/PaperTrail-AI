import pytest
from unittest.mock import MagicMock
from backend.app.schemas import Chunk, ChunkKind
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.hybrid import HybridRetriever, select_allowed_candidates
from backend.app.eval.run_retrieval import build_eval_rrf_pool


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


def test_hybrid_retriever_empty_corpus():
    """HybridRetriever search on empty un-indexed corpus returns [] gracefully."""
    retriever = HybridRetriever()
    results = retriever.search("query on empty corpus", top_k=5)
    assert results == []


def test_hybrid_retriever_out_of_vocabulary():
    """HybridRetriever handles queries with completely out-of-vocabulary tokens."""
    retriever = HybridRetriever()
    corpus = [
        Chunk(
            chunk_id="chk-v1",
            doc_id="doc-1",
            page=1,
            section_heading="Biology",
            text="Cellular mitosis and meiosis in eukaryotic organisms.",
            kind=ChunkKind.TEXT,
        )
    ]
    retriever.index_chunks(corpus)

    # Query with non-existent words (absent from BM25)
    results = retriever.search("zzzzquark zzzzgluon", top_k=5)
    # Dense semantic search still provides embedding distance, BM25 rank is None
    assert len(results) == 1
    assert results[0].chunk_id == "chk-v1"
    assert results[0].bm25_rank is None
    assert results[0].dense_rank == 1


def test_hybrid_retriever_duplicates_and_small_corpus():
    """
    HybridRetriever handles small corpora (smaller than candidate pool)
    and deduplicates candidate pool by chunk ID cleanly.
    """
    retriever = HybridRetriever()
    small_corpus = [
        Chunk(
            chunk_id="chunk-single",
            doc_id="doc-small",
            page=1,
            section_heading="Small Intro",
            text="A single standalone passage in a minimal corpus.",
            kind=ChunkKind.TEXT,
        )
    ]
    retriever.index_chunks(small_corpus)

    # Search with candidate_pool_size=30 on 1-chunk corpus
    results = retriever.search("minimal corpus", top_k=10, candidate_pool_size=30)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-single"
    assert results[0].fused_rank == 1
    assert retriever.last_total_candidates == 1


def test_candidate_selection_alignment_prevents_dilution():
    """
    Verify that candidate selection in HybridRetriever and the evaluation runner
    caps both modalities to candidate_pool_size before RRF, preventing dilution
    where dense distractors push a strong BM25 rank-3 hit past candidate_pool_size.
    """
    retriever = HybridRetriever()
    target_chunk = Chunk(
        chunk_id="chk-target-solver",
        doc_id="doc-1",
        page=6,
        section_heading="Implementation Details",
        text="For the ILP search we use the Gurobi solver.",
        kind=ChunkKind.TEXT,
    )
    bm25_top1 = Chunk(chunk_id="chk-bm25-1", doc_id="doc-1", page=5, text="top1", kind=ChunkKind.TEXT)
    bm25_top2 = Chunk(chunk_id="chk-bm25-2", doc_id="doc-1", page=5, text="top2", kind=ChunkKind.TEXT)

    # 60 distractors representing full retrieval depth (candidate_pool_size * 2 = 60)
    distractors = [
        Chunk(chunk_id=f"chk-dense-{i}", doc_id="doc-1", page=i + 1, text=f"distractor {i}", kind=ChunkKind.TEXT)
        for i in range(60)
    ]
    all_chunks = [target_chunk, bm25_top1, bm25_top2] + distractors
    retriever.chunks_by_id = {c.chunk_id: c for c in all_chunks}

    # BM25 ranking (38 items returned, consecutive ranks 1..38): top1, top2, target (rank 3), 10 distractors (ranks 4..13), 25 distractors (ranks 14..38)
    bm25_mock_results = [
        (bm25_top1, 20.0, 1),
        (bm25_top2, 18.0, 2),
        (target_chunk, 15.0, 3),
    ]
    bm25_mock_results += [(distractors[i], 12.0 - i * 0.1, i + 4) for i in range(10)]
    bm25_mock_results += [(distractors[i], 5.0 - i * 0.1, i + 4) for i in range(10, 35)]
    retriever.bm25.search = MagicMock(return_value=bm25_mock_results)

    # Dense ranking (60 items returned): all 60 distractors have ranks 1..60, target is absent (>60)
    dense_mock_results = [(distractors[i], 0.9 - i * 0.01, i + 1) for i in range(60)]
    retriever.dense.search = MagicMock(return_value=dense_mock_results)

    # Live retriever search with candidate_pool_size=30 (which searches top_k=60 from each modality)
    live_results = retriever.search("query", top_k=30, candidate_pool_size=30, use_reranker=False)
    live_cids = [r.chunk_id for r in live_results]

    # 1. Target must be included in the live candidate pool (Rank <= 30)
    assert "chk-target-solver" in live_cids, (
        f"Target chunk was excluded from live candidate pool of size 30. Returned: {live_cids}"
    )

    # 2. Evaluation runner candidate selection via production build_eval_rrf_pool:
    _, eval_pool_chunks = build_eval_rrf_pool(
        bm25_mock_results, dense_mock_results, retriever.chunks_by_id, pool_size=30
    )
    eval_pool_cids = [c.chunk_id for c in eval_pool_chunks]

    # 3. Assert live retriever candidate pool and evaluation runner candidate pool are identical
    assert live_cids == eval_pool_cids

    # 4. Old Live Behavior: all 60 candidates retrieved from each modality were fused without pre-capping
    old_live_fused = reciprocal_rank_fusion(
        [
            [(c.chunk_id, r) for c, _, r in bm25_mock_results],
            [(c.chunk_id, r) for c, _, r in dense_mock_results],
        ],
        k=60,
    )
    old_live_pool_cids = [cid for cid, _ in old_live_fused[:30]]
    assert "chk-target-solver" not in old_live_pool_cids, (
        "Under old live behavior, target chunk should be diluted past candidate_pool_size (30)."
    )


def test_candidate_selection_with_filtered_references_depth_consistency():
    """
    Verify that when raw retrieval results contain ChunkKind.REFERENCES:
    1. The shared candidate selection helper (select_allowed_candidates) filters references
       while strictly preserving the original retrieval ranks.
    2. Both live retrieval (HybridRetriever) and the evaluation runner (build_eval_rrf_pool)
       use the shared helper and retrieve at depth candidate_pool_size * 2 (60),
       producing identical eligible candidate lists and the exact same fused candidate pool.
    3. Under the old evaluation depth (50), reference filtering resulted in candidate starvation (only 25 chunks).
    """
    retriever = HybridRetriever()

    # 25 reference chunks that rank high in BM25 (ranks 1..25)
    ref_chunks = [
        Chunk(chunk_id=f"ref-{i}", doc_id="doc-1", page=10, text=f"Bibliography entry {i}", kind=ChunkKind.REFERENCES)
        for i in range(25)
    ]
    # 35 legitimate text chunks that rank lower in BM25 (ranks 26..60)
    text_chunks = [
        Chunk(chunk_id=f"text-{i}", doc_id="doc-1", page=1, text=f"Main content {i}", kind=ChunkKind.TEXT)
        for i in range(35)
    ]
    all_chunks = ref_chunks + text_chunks
    chunk_map = {c.chunk_id: c for c in all_chunks}
    retriever.chunks_by_id = chunk_map

    # BM25 returns 25 references followed by 35 text chunks (total 60)
    bm25_raw = [(ref_chunks[i], 50.0 - i, i + 1) for i in range(25)]
    bm25_raw += [(text_chunks[i], 20.0 - i, i + 26) for i in range(35)]
    retriever.bm25.search = MagicMock(return_value=bm25_raw)

    # Dense returns 35 text chunks (ranks 1..35)
    dense_raw = [(text_chunks[i], 0.9 - i * 0.01, i + 1) for i in range(35)]
    retriever.dense.search = MagicMock(return_value=dense_raw)

    # 1. Test shared selection helper: eligible candidate lists and rank preservation
    bm25_eligible = select_allowed_candidates(bm25_raw, max_candidates=30)
    dense_eligible = select_allowed_candidates(dense_raw, max_candidates=30)

    assert len(bm25_eligible) == 30
    assert not any(c.kind == ChunkKind.REFERENCES for c, _, _ in bm25_eligible)
    # Original retrieval ranks must be preserved when filtering references:
    # First allowed text chunk was at rank 26 in bm25_raw; 30th allowed text chunk was at rank 55
    assert bm25_eligible[0][2] == 26
    assert bm25_eligible[-1][2] == 55
    assert [r for _, _, r in bm25_eligible] == list(range(26, 56))

    assert len(dense_eligible) == 30
    assert [r for _, _, r in dense_eligible] == list(range(1, 31))

    # 2. Live retrieval execution (HybridRetriever)
    live_results = retriever.search("query", top_k=30, candidate_pool_size=30, use_reranker=False)
    live_cids = [r.chunk_id for r in live_results]

    # No references must leak into the candidate pool
    assert not any(cid.startswith("ref-") for cid in live_cids)
    assert len(live_cids) == 30
    # Specifically, text-25 through text-29 (raw BM25 ranks 51..55) must be present in the pool
    assert "text-25" in live_cids
    assert "text-29" in live_cids

    # 3. Production evaluation runner execution (build_eval_rrf_pool)
    _, eval_pool_chunks = build_eval_rrf_pool(
        bm25_raw=bm25_raw,
        dense_raw=dense_raw,
        chunk_map=chunk_map,
        pool_size=30,
    )
    eval_pool_cids = [c.chunk_id for c in eval_pool_chunks]

    # Assert live retrieval and evaluation runner select the exact same candidate pool
    assert live_cids == eval_pool_cids

    # 4. Prove that under old evaluation depth (50),
    # text-25 through text-29 were excluded because only 25 allowed chunks existed in top 50
    old_eval_bm25_allowed = select_allowed_candidates(bm25_raw, max_candidates=30, raw_depth=50)
    assert len(old_eval_bm25_allowed) == 25  # starved to 25
    assert "text-25" not in [c.chunk_id for c, _, _ in old_eval_bm25_allowed]


def test_select_allowed_candidates_unit():
    """
    Direct unit tests for select_allowed_candidates helper:
    - reference filtering when include_references=False
    - reference inclusion when include_references=True
    - document ID filtering
    - preserving original retrieval ranks
    - respecting max_candidates limit
    """
    c_text1 = Chunk(chunk_id="c1", doc_id="docA", page=1, text="Text 1", kind=ChunkKind.TEXT)
    c_ref1 = Chunk(chunk_id="r1", doc_id="docA", page=10, text="Ref 1", kind=ChunkKind.REFERENCES)
    c_text2 = Chunk(chunk_id="c2", doc_id="docB", page=2, text="Text 2", kind=ChunkKind.TEXT)
    c_text3 = Chunk(chunk_id="c3", doc_id="docA", page=3, text="Text 3", kind=ChunkKind.TEXT)

    raw_results = [
        (c_text1, 10.0, 1),
        (c_ref1, 9.0, 2),
        (c_text2, 8.0, 3),
        (c_text3, 7.0, 4),
    ]

    # By default, exclude references
    res_no_ref = select_allowed_candidates(raw_results, max_candidates=10)
    assert [c.chunk_id for c, _, _ in res_no_ref] == ["c1", "c2", "c3"]
    assert [r for _, _, r in res_no_ref] == [1, 3, 4]

    # When include_references=True, include reference chunk
    res_with_ref = select_allowed_candidates(raw_results, max_candidates=10, include_references=True)
    assert [c.chunk_id for c, _, _ in res_with_ref] == ["c1", "r1", "c2", "c3"]
    assert [r for _, _, r in res_with_ref] == [1, 2, 3, 4]

    # Filtering by doc_ids: matching ID, non-matching ID, None (all allowed), and empty set (none allowed)
    res_none = select_allowed_candidates(raw_results, max_candidates=10, doc_ids=None)
    assert [c.chunk_id for c, _, _ in res_none] == ["c1", "c2", "c3"]

    res_empty_set = select_allowed_candidates(raw_results, max_candidates=10, doc_ids=set())
    assert res_empty_set == []

    res_matching = select_allowed_candidates(raw_results, max_candidates=10, doc_ids={"docA"})
    assert [c.chunk_id for c, _, _ in res_matching] == ["c1", "c3"]
    assert [r for _, _, r in res_matching] == [1, 4]

    res_non_matching = select_allowed_candidates(raw_results, max_candidates=10, doc_ids={"docNonExistent"})
    assert res_non_matching == []

    # Capping at max_candidates
    res_capped = select_allowed_candidates(raw_results, max_candidates=2)
    assert len(res_capped) == 2
    assert [c.chunk_id for c, _, _ in res_capped] == ["c1", "c2"]


def test_hybrid_retriever_doc_id_filter_semantics(sample_chunks):
    """
    Verify live retrieval (HybridRetriever.search) document filter semantics:
    - doc_ids=None: searches across all documents in corpus
    - doc_ids=[] (empty list): no documents allowed, returns empty list
    - doc_ids=set() (empty set): no documents allowed, returns empty list
    - doc_ids=['doc-1'] (matching): returns only chunks from doc-1
    - doc_ids=['doc-nonexistent'] (non-matching): returns empty list
    """
    retriever = HybridRetriever()
    retriever.index_chunks(sample_chunks)

    # 1. doc_ids=None -> all documents allowed
    res_none = retriever.search("Transformer BLEU", doc_ids=None)
    assert len(res_none) > 0
    assert any(r.doc_id == "doc-1" for r in res_none)

    # 2. doc_ids=[] -> no documents allowed
    res_empty_list = retriever.search("Transformer BLEU", doc_ids=[])
    assert res_empty_list == []
    assert retriever.last_total_candidates == 0

    # 3. doc_ids=set() -> no documents allowed
    res_empty_set = retriever.search("Transformer BLEU", doc_ids=set())
    assert res_empty_set == []
    assert retriever.last_total_candidates == 0

    # 4. doc_ids=['doc-1'] -> only chunks matching doc-1
    res_matching = retriever.search("Transformer BLEU", doc_ids=["doc-1"])
    assert len(res_matching) > 0
    assert all(r.doc_id == "doc-1" for r in res_matching)

    # 5. doc_ids=['doc-nonexistent'] -> non-matching ID, returns empty list
    res_non_matching = retriever.search("Transformer BLEU", doc_ids=["doc-nonexistent"])
    assert res_non_matching == []
    assert retriever.last_total_candidates == 0



def test_get_methodological_context_deterministic_selection():
    """Verify deterministic selection of bounded method context from the same document."""
    from backend.app.retrieval.rerank import get_methodological_context

    method_chunk = Chunk(
        chunk_id="chk-m1",
        doc_id="doc-xyz",
        page=4,
        section_heading="3. Methodology and Algorithmic Formulation",
        text="We formulate the parameter allocation problem as an integer linear program solved via off-the-shelf solvers.",
        kind=ChunkKind.TEXT,
    )
    eval_chunk = Chunk(
        chunk_id="chk-e1",
        doc_id="doc-xyz",
        page=5,
        section_heading="4. Evaluation and Implementation",
        text="For the ILP search we use the Gurobi optimizer version 11.",
        kind=ChunkKind.TEXT,
    )
    doc_chunks = [method_chunk, eval_chunk]

    ctx = get_methodological_context(eval_chunk, doc_chunks=doc_chunks, max_context_chars=100)
    assert ctx.startswith("\n[Method Context: 3. Methodology and Algorithmic Formulation - ")
    assert "solved via off-the-shelf solvers." in ctx
    # Non-evaluation chunk gets empty context
    assert get_methodological_context(method_chunk, doc_chunks=doc_chunks) == ""


def test_get_methodological_context_safeguards():
    """Verify context isolation, skipping of tables/references, and malformed metadata resilience."""
    from backend.app.retrieval.rerank import get_methodological_context

    unrelated_method = Chunk(
        chunk_id="chk-other-doc",
        doc_id="doc-DIFFERENT",
        page=1,
        section_heading="3. Algorithm",
        text="Unrelated algorithm details.",
        kind=ChunkKind.TEXT,
    )
    table_chunk = Chunk(
        chunk_id="chk-tbl",
        doc_id="doc-xyz",
        page=4,
        section_heading="3. Architecture",
        text="Table 1: Benchmark results on baseline models.",
        kind=ChunkKind.TABLE,
    )
    ref_chunk = Chunk(
        chunk_id="chk-ref",
        doc_id="doc-xyz",
        page=4,
        section_heading="3. Architecture",
        text="[1] References and bibliography.",
        kind=ChunkKind.REFERENCES,
    )
    eval_chunk = Chunk(
        chunk_id="chk-e2",
        doc_id="doc-xyz",
        page=5,
        section_heading="5. Experimental Setup",
        text="We run all experiments on an H200 GPU.",
        kind=ChunkKind.TEXT,
    )

    # 1. Unrelated document chunks are never attached
    doc_chunks_with_unrelated = [unrelated_method, eval_chunk]
    assert get_methodological_context(eval_chunk, doc_chunks=doc_chunks_with_unrelated) == ""

    # 2. Table and references chunks are skipped
    doc_chunks_with_table_ref = [table_chunk, ref_chunk, eval_chunk]
    assert get_methodological_context(eval_chunk, doc_chunks=doc_chunks_with_table_ref) == ""

    # 3. Missing or empty metadata does not crash
    eval_empty_heading = Chunk(
        chunk_id="chk-e3",
        doc_id="doc-xyz",
        page=5,
        section_heading=None,
        text="Some text without heading.",
        kind=ChunkKind.TEXT,
    )
    assert get_methodological_context(eval_empty_heading, doc_chunks=[]) == ""
    assert get_methodological_context(eval_chunk, doc_chunks=[]) == ""
    assert get_methodological_context(eval_chunk, doc_chunks=None) == ""


def test_cross_encoder_reranker_preserves_identities_and_input_order():
    """Verify that CrossEncoder receives [query, text + context] and preserves chunk identity."""
    from backend.app.retrieval.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.85, 0.42]
    reranker._model = mock_model

    c1 = Chunk(
        chunk_id="chk-orig-1",
        doc_id="doc-test",
        page=7,
        section_heading="4. Evaluation",
        text="Original passage text 1.",
        kind=ChunkKind.TEXT,
    )
    c2 = Chunk(
        chunk_id="chk-orig-2",
        doc_id="doc-test",
        page=8,
        section_heading="2. Theory",
        text="Original passage text 2.",
        kind=ChunkKind.TEXT,
    )

    query = "Which solver is used?"
    results = reranker.rerank(query, [c1, c2], top_k=2)

    # 1. Returned Chunk objects preserve exact identities and raw text
    assert len(results) == 2
    r_chunk_1, score_1, rank_1 = results[0]
    assert r_chunk_1.chunk_id == "chk-orig-1"
    assert r_chunk_1.doc_id == "doc-test"
    assert r_chunk_1.page == 7
    assert r_chunk_1.text == "Original passage text 1."
    assert rank_1 == 1
    assert score_1 == 0.85

    # 2. Verify query-passage pair ordering passed to model.predict
    assert mock_model.predict.called
    pairs_passed = mock_model.predict.call_args[0][0]
    assert len(pairs_passed) == 2
    assert pairs_passed[0][0] == query  # Query is first element
    assert pairs_passed[0][1].startswith("4. Evaluation\nOriginal passage text 1.")
    assert pairs_passed[1][0] == query


def test_q012_reranking_fix_reproduction():
    """
    End-to-end integration test:
    Verify that Q-012 gold chunk doc_c67573ef50cc_p6_c2 reaches top 5 in hybrid search
    with its original chunk_id, page, and exact quote preserved.
    """
    import json
    from pathlib import Path
    from backend.app.core.storage import storage as current_storage
    from backend.app.schemas import Document
    from backend.app.retrieval.hybrid import retriever

    # Load corpus chunks from repository data into test storage if not present
    chunks_path = Path("data/chunks/doc_c67573ef50cc_chunks.json")
    doc_path = Path("data/documents/doc_c67573ef50cc.json")
    if chunks_path.exists() and doc_path.exists():
        with open(doc_path, "r", encoding="utf-8") as df:
            doc = Document(**json.load(df))
        with open(chunks_path, "r", encoding="utf-8") as cf:
            chunks = [Chunk(**cd) for cd in json.load(cf)]
        current_storage.save_document(doc, chunks, force_overwrite=True)
        retriever.index_chunks(chunks)

    query = "Which off-the-shelf solver does LEMS use for its allocation problem?"
    results = retriever.search(query, top_k=5)

    assert len(results) >= 3
    top_5_ids = [r.chunk_id for r in results]
    assert "doc_c67573ef50cc_p6_c2" in top_5_ids

    # Find the gold chunk result
    gold_result = next(r for r in results if r.chunk_id == "doc_c67573ef50cc_p6_c2")
    assert gold_result.page == 6
    assert gold_result.doc_id == "doc_c67573ef50cc"
    assert "we use the Gurobi solver" in gold_result.text
    assert gold_result.rerank_rank <= 5
