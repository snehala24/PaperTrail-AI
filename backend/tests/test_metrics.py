import pytest
from backend.app.schemas import Chunk, ChunkKind, GoldEvidence, EvidenceRelation
from backend.app.eval.metrics import compute_retrieval_metrics


def test_compute_retrieval_metrics():
    gold = [
        GoldEvidence(
            gold_id="g1",
            question_id="q1",
            doc_sha256="doc-1-sha",
            page=2,
            answer_span="28.4 BLEU",
            expected_relation=EvidenceRelation.SUPPORTS,
        )
    ]

    # Candidate list where rank 1 matches the gold span on page 2
    retrieved_chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc-1",
            page=2,
            text="The base Transformer model achieved 28.4 BLEU on the test set.",
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="c2",
            doc_id="doc-1",
            page=1,
            text="Unrelated introductory text.",
            kind=ChunkKind.TEXT,
        ),
    ]

    metrics = compute_retrieval_metrics(retrieved_chunks, gold, k_list=[1, 5])

    # Rank 1 is relevant
    assert metrics["recall@1"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg@1"] == 1.0
    assert metrics["ndcg@5"] == 1.0


def test_compute_retrieval_metrics_rank2_hit():
    gold = [
        GoldEvidence(
            gold_id="g1",
            question_id="q1",
            doc_sha256="doc-1-sha",
            page=2,
            answer_span="28.4 BLEU",
            expected_relation=EvidenceRelation.SUPPORTS,
        )
    ]

    # Rank 1 is non-relevant, Rank 2 matches
    retrieved_chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc-1",
            page=1,
            text="Unrelated text.",
            kind=ChunkKind.TEXT,
        ),
        Chunk(
            chunk_id="c2",
            doc_id="doc-1",
            page=2,
            text="The base Transformer model achieved 28.4 BLEU on the test set.",
            kind=ChunkKind.TEXT,
        ),
    ]

    metrics = compute_retrieval_metrics(retrieved_chunks, gold, k_list=[1, 2])

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 1.0
    assert metrics["precision@1"] == 0.0
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr"] == 0.5  # 1 / rank 2
    assert metrics["ndcg@1"] == 0.0
    assert metrics["ndcg@2"] > 0.0
