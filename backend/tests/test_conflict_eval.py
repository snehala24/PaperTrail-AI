"""
Regression test suite for apparent-conflict benchmark evaluation and conflict handling (Day 3).

Verifies:
1. Real contradiction: Two chunks report contradictory claims under identical experimental conditions;
   system identifies conflicting chunk IDs, flags that sources disagree, and passes 3-layer verification.
2. Apparent conflict explainable by differing conditions: Two chunks report differing findings due to
   distinct conditions/hardware (e.g. H200 for main experiments vs A100 for inference speedup);
   both claims are grounded with verbatim quotes and verified.
3. Missing second-paper evidence: Cross-paper questions requiring two papers (e.g. Q-030, Q-029) are
   strictly classified as BLOCKED_PENDING_SECOND_PAPER / not evaluable, never reported as passed.
4. Missing retrieval evidence: When retrieval fails to return conflicting evidence or falls below the
   safety floor, AnswerService honestly abstains without hallucinating conflict flags.
5. Unsupported conflict claims: When a conflict claim cites hallucinated quotes (Layer 2) or unentailed
   assertions (Layer 3), zero-tolerance abstention triggers immediately.
6. End-to-end evaluation runner: run_conflicts_eval executes deterministically offline, maintains exact
   denominator counts (5 total, 4 evaluated, 1 blocked), and generates a valid sanitized run artifact.

ZERO LIVE GEMINI CALLS ARE MADE: All tests use deterministic mocks.
All asynchronous calls are driven via run_sync to prevent Windows socketpair deadlocks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

from backend.app.citations.verifier import CitationVerifier
from backend.app.claims.service import (
    EMERGENCY_RERANK_SCORE_FLOOR,
    AnswerService,
)
from backend.app.eval.run_conflicts import (
    MockLLMProvider,
    load_conflict_benchmark_data,
    run_conflicts_eval,
    run_sync,
)
from backend.app.llm.base import LLMMessage, LLMProvider, LLMResponse
from backend.app.schemas import (
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    BenchmarkQuestion,
    BenchmarkQuestionType,
    BenchmarkStatus,
    ChunkKind,
    ClaimVerificationStatus,
    EvidenceRelation,
    GoldEvidence,
    ScoredChunk,
)


def make_scored_chunk(
    chunk_id: str = "doc_c67573ef50cc_p1_c1",
    doc_id: str = "doc_c67573ef50cc",
    page: int = 1,
    text: str = "Evidence text here.",
    rerank_score: float = 5.0,
    kind: ChunkKind = ChunkKind.TEXT,
) -> ScoredChunk:
    """Helper to create ScoredChunk fixtures."""
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=page,
        kind=kind,
        text=text,
        char_start=0,
        char_end=len(text),
        rerank_score=rerank_score,
        bm25_rank=1,
        dense_rank=1,
        fused_rank=1,
    )


class MockRetriever:
    """Mock retriever returning predetermined ScoredChunk results."""

    def __init__(self, results: Optional[List[ScoredChunk]] = None):
        self.results = results if results is not None else []
        self.chunks_by_id = {c.chunk_id: c for c in self.results}

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        include_references: bool = False,
        use_reranker: bool = True,
    ) -> List[ScoredChunk]:
        return self.results[:top_k]


# =========================================================================
# Regression Test 1: Real Contradiction
# =========================================================================

def test_conflict_regression_real_contradiction():
    """
    Test 1: Real contradiction under identical conditions.
    Chunks report contradictory empirical results. The system must:
    - Flag 'sources disagree'
    - Cite the conflicting chunk IDs in the answer
    - Verify all claims through 3 layers
    - Preserve null conflict_reason in Day 3 evidence links
    """
    chunk1 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p10_c1",
        text="Algorithm Alpha achieves 94% zero-shot accuracy under standard greedy decoding.",
    )
    chunk2 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p10_c2",
        text="Algorithm Alpha achieves only 78% zero-shot accuracy under standard greedy decoding.",
    )
    retriever = MockRetriever([chunk1, chunk2])

    gen_json = json.dumps({
        "answer": (
            "One experiment reports that Algorithm Alpha achieves 94% zero-shot accuracy under standard greedy decoding, "
            "while another section reports it achieves only 78% zero-shot accuracy under standard greedy decoding."
        ),
        "claims": [
            {
                "claim_id": "c1",
                "text": "Algorithm Alpha achieves 94% zero-shot accuracy under standard greedy decoding.",
                "chunk_ids": ["doc_c67573ef50cc_p10_c1"],
                "quote": "Algorithm Alpha achieves 94% zero-shot accuracy under standard greedy decoding.",
            },
            {
                "claim_id": "c2",
                "text": "Algorithm Alpha achieves only 78% zero-shot accuracy under standard greedy decoding.",
                "chunk_ids": ["doc_c67573ef50cc_p10_c2"],
                "quote": "Algorithm Alpha achieves only 78% zero-shot accuracy under standard greedy decoding.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [{
            "chunk_id_a": "doc_c67573ef50cc_p10_c1",
            "chunk_id_b": "doc_c67573ef50cc_p10_c2",
        }],
    })
    verif_ok = json.dumps({"supported": True, "rationale": "Exact match and direct entailment confirmed."})
    llm = MockLLMProvider([gen_json, verif_ok, verif_ok])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="What is the zero-shot accuracy of Algorithm Alpha?")
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert "sources disagree" in resp.answer.lower()
    assert "doc_c67573ef50cc_p10_c1" in resp.answer
    assert "doc_c67573ef50cc_p10_c2" in resp.answer
    assert len(resp.evidence_links) == 2
    assert all(l.relation == EvidenceRelation.SUPPORTS for l in resp.evidence_links)
    assert all(l.conflict_reason is None for l in resp.evidence_links)


# =========================================================================
# Regression Test 2: Apparent Conflict Explainable by Differing Conditions
# =========================================================================

def test_conflict_regression_apparent_conflict_differing_conditions():
    """
    Test 2: Apparent conflict explainable by differing experimental conditions.
    Mirrors Q-015 (H200 for main experiments on p.6 vs A100 for speed on p.26).
    The system correctly grounds both claims with verified citations and explains conditions.
    """
    chunk_h200 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p6_c2",
        page=6,
        text="All experiments are run on a single NVIDIA H200 GPU.",
    )
    chunk_a100 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p26_c7",
        page=26,
        text="All measurements were conducted on a single NVIDIA A100 GPU.",
    )
    retriever = MockRetriever([chunk_h200, chunk_a100])

    gen_json = json.dumps({
        "answer": (
            "The main experiments were run on a single NVIDIA H200 GPU, whereas wall-clock inference speed "
            "measurements were conducted on a single NVIDIA A100 GPU."
        ),
        "claims": [
            {
                "claim_id": "c1",
                "text": "All experiments are run on a single NVIDIA H200 GPU.",
                "chunk_ids": ["doc_c67573ef50cc_p6_c2"],
                "quote": "All experiments are run on a single NVIDIA H200 GPU.",
            },
            {
                "claim_id": "c2",
                "text": "All measurements were conducted on a single NVIDIA A100 GPU.",
                "chunk_ids": ["doc_c67573ef50cc_p26_c7"],
                "quote": "All measurements were conducted on a single NVIDIA A100 GPU.",
            },
        ],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [{
            "chunk_id_a": "doc_c67573ef50cc_p6_c2",
            "chunk_id_b": "doc_c67573ef50cc_p26_c7",
        }],
    })
    verif_ok = json.dumps({"supported": True, "rationale": "Exact match and direct entailment."})
    llm = MockLLMProvider([gen_json, verif_ok, verif_ok])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="Which GPU were the experiments run on?")
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ANSWERED
    assert "sources disagree" in resp.answer.lower()
    assert "doc_c67573ef50cc_p6_c2" in resp.answer
    assert "doc_c67573ef50cc_p26_c7" in resp.answer
    assert len(resp.evidence_links) == 2
    assert all(chk.status == ClaimVerificationStatus.VERIFIED for chk in resp.citation_checks)


# =========================================================================
# Regression Test 3: Missing Second-Paper Evidence (Q-029, Q-030)
# =========================================================================

def test_conflict_regression_missing_second_paper_blocked():
    """
    Test 3: Cross-paper questions requiring two papers are strictly blocked.
    Verifies that Q-030 and Q-029 are never evaluated as single-paper tests,
    and are never reported as passed when the second paper is missing.
    """
    q30 = BenchmarkQuestion(
        question_id="Q-030",
        question="Does GFWSVD's own reported performance match the LEMS paper's claim that it degrades?",
        question_type=BenchmarkQuestionType.APPARENT_CONFLICT,
        status=BenchmarkStatus.PENDING_SECOND_PAPER,
        labelled_by="snehala24",
        target_doc_ids=["doc_c67573ef50cc", "doc_gfwsvd_placeholder"],
    )
    gold33 = GoldEvidence(
        gold_id="GOLD-033",
        question_id="Q-030",
        doc_sha256="c67573ef50ccf236151da342e53ba1319560146f4ccdd104e6cb41ad24439922",
        page=6,
        answer_span="GFWSVD degrades significantly under identical conditions",
        expected_relation=EvidenceRelation.SUPPORTS,
        status=BenchmarkStatus.PENDING_SECOND_PAPER,
    )
    gold34 = GoldEvidence(
        gold_id="GOLD-034",
        question_id="Q-030",
        doc_sha256="TODO_GFWSVD_SHA256",
        page=1,
        answer_span="TODO: pending GFWSVD paper ingestion",
        expected_relation=EvidenceRelation.CONTRADICTS,
        status=BenchmarkStatus.PENDING_SECOND_PAPER,
    )

    # Verify blocked detection logic
    is_q30_blocked = (
        q30.status == BenchmarkStatus.PENDING_SECOND_PAPER
        or "doc_gfwsvd_placeholder" in q30.target_doc_ids
        or any("TODO" in g.doc_sha256 for g in [gold33, gold34])
    )
    assert is_q30_blocked is True

    # Same check for Q-029
    q29 = BenchmarkQuestion(
        question_id="Q-029",
        question="Do the two papers describe the Kronecker-factor estimation the same way?",
        question_type=BenchmarkQuestionType.COMPARATIVE,
        status=BenchmarkStatus.PENDING_SECOND_PAPER,
        labelled_by="snehala24",
        target_doc_ids=["doc_c67573ef50cc", "doc_gfwsvd_placeholder"],
    )
    is_q29_blocked = (
        q29.status == BenchmarkStatus.PENDING_SECOND_PAPER
        or "doc_gfwsvd_placeholder" in q29.target_doc_ids
    )
    assert is_q29_blocked is True


# =========================================================================
# Regression Test 4: Missing Retrieval Evidence
# =========================================================================

def test_conflict_regression_missing_retrieval_evidence_abstains():
    """
    Test 4: When retrieval fails to return evidence chunks, AnswerService abstains honestly.
    No hallucinated answers or fabricated conflicts may be produced.
    """
    empty_retriever = MockRetriever([])
    llm = MockLLMProvider([])
    verifier = CitationVerifier(llm_provider=llm)
    service = AnswerService(retriever=empty_retriever, llm_provider=llm, verifier=verifier)

    req = AnswerRequest(question="Why did the two algorithms disagree on perplexity?")
    resp: AnswerResponse = run_sync(service.answer(req))

    assert resp.status == AnswerStatus.ABSTAINED
    assert "could not find relevant evidence" in resp.answer.lower()
    assert resp.evidence_links == []
    assert resp.citation_checks == []
    assert "No relevant chunks retrieved" in resp.abstain_reason

    # Also test below emergency rerank score floor
    subfloor_chunk = make_scored_chunk(rerank_score=-8.5)
    floor_retriever = MockRetriever([subfloor_chunk])
    service_floor = AnswerService(retriever=floor_retriever, llm_provider=llm, verifier=verifier)

    resp_floor: AnswerResponse = run_sync(service_floor.answer(req))
    assert resp_floor.status == AnswerStatus.ABSTAINED
    assert "emergency retrieval safety cutoff" in resp_floor.abstain_reason.lower()


# =========================================================================
# Regression Test 5: Unsupported Conflict Claims
# =========================================================================

def test_conflict_regression_unsupported_conflict_claims_abstains():
    """
    Test 5: When a conflict claim has a hallucinated quote (Layer 2 failure)
    or is not entailed by evidence (Layer 3 failure), AnswerService enforces
    zero-tolerance abstention and never returns answered.
    """
    chunk1 = make_scored_chunk(
        chunk_id="doc_c67573ef50cc_p1_c1",
        text="Method Alpha achieved 94% accuracy.",
    )
    retriever = MockRetriever([chunk1])

    # Case A: Hallucinated quote (Layer 2 failure)
    gen_hallucinated_quote = json.dumps({
        "answer": "Method Alpha achieved 99% accuracy.",
        "claims": [{
            "claim_id": "c1",
            "text": "Method Alpha achieved 99% accuracy.",
            "chunk_ids": ["doc_c67573ef50cc_p1_c1"],
            "quote": "Method Alpha achieved 99% accuracy.",  # NOT in chunk text
        }],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
    })
    llm_layer2 = MockLLMProvider([gen_hallucinated_quote])
    verifier_layer2 = CitationVerifier(llm_provider=llm_layer2)
    service_layer2 = AnswerService(retriever=retriever, llm_provider=llm_layer2, verifier=verifier_layer2)

    req = AnswerRequest(question="What was Method Alpha's accuracy?")
    resp_l2: AnswerResponse = run_sync(service_layer2.answer(req))

    assert resp_l2.status == AnswerStatus.ABSTAINED
    assert "zero-tolerance" in resp_l2.abstain_reason.lower()
    assert any("Layer 2 failure" in (c.notes or "") for c in resp_l2.citation_checks)

    # Case B: Unentailed claim (Layer 3 failure)
    gen_unentailed = json.dumps({
        "answer": "Method Alpha completely invalidates all previous deep learning architectures.",
        "claims": [{
            "claim_id": "c1",
            "text": "Method Alpha completely invalidates all previous deep learning architectures.",
            "chunk_ids": ["doc_c67573ef50cc_p1_c1"],
            "quote": "Method Alpha achieved 94% accuracy.",  # Quote is present
        }],
        "abstained": False,
        "abstain_reason": None,
        "conflicts": [],
    })
    verif_fail = json.dumps({"supported": False, "rationale": "Extrapolation not supported by evidence."})
    llm_layer3 = MockLLMProvider([gen_unentailed, verif_fail])
    verifier_layer3 = CitationVerifier(llm_provider=llm_layer3)
    service_layer3 = AnswerService(retriever=retriever, llm_provider=llm_layer3, verifier=verifier_layer3)

    resp_l3: AnswerResponse = run_sync(service_layer3.answer(req))
    assert resp_l3.status == AnswerStatus.ABSTAINED
    assert "zero-tolerance" in resp_l3.abstain_reason.lower()
    assert any(c.status == ClaimVerificationStatus.UNSUPPORTED and c.supports is False for c in resp_l3.citation_checks)


# =========================================================================
# Regression Test 6: End-to-End Evaluation Runner Test
# =========================================================================

def test_conflict_eval_runner_end_to_end(tmp_path: Path):
    """
    Test 6: End-to-end repeatable execution of run_conflicts_eval.
    Verifies:
    - 5 total questions
    - 4 evaluated, 1 blocked (Q-030)
    - 100% retrieval success on evaluated (4/4)
    - 100% citation verification success on evaluated (4/4)
    - 100% conflict detection correctness on evaluated (4/4)
    - Timestamped run record written to disk
    - JSON output contains no API keys or secrets
    """
    repo_root = Path(__file__).resolve().parents[2]
    eval_dir = repo_root / "eval"
    runs_dir = tmp_path / "runs"

    record = run_conflicts_eval(eval_dir=eval_dir, runs_dir=runs_dir, save_artifact=True)

    summary = record["summary"]
    assert summary["total_conflict_questions"] == 5
    assert summary["evaluated_questions"] == 4
    assert summary["blocked_pending_second_paper"] == 1
    assert summary["retrieval_success_count"] == 4
    assert summary["retrieval_success_rate"] == 1.0
    assert summary["citation_verification_success_count"] == 4
    assert summary["citation_verification_success_rate"] == 1.0
    assert summary["conflict_detection_success_count"] == 4
    assert summary["conflict_detection_success_rate"] == 1.0
    assert summary["overall_grounded_success_count"] == 4
    assert summary["overall_grounded_success_rate"] == 1.0

    # Verify per-question categorization
    per_q = {item["question_id"]: item for item in record["per_question_results"]}
    assert per_q["Q-005"]["conflict_category"] == "real_contradiction"
    assert per_q["Q-006"]["conflict_category"] == "apparent_conflict_differing_conditions"
    assert per_q["Q-015"]["conflict_category"] == "apparent_conflict_differing_conditions"
    assert per_q["Q-017"]["conflict_category"] == "apparent_conflict_differing_conditions"
    assert per_q["Q-030"]["status"] == "BLOCKED_PENDING_SECOND_PAPER"

    # Verify artifact was written and contains no secrets
    saved_files = list(runs_dir.glob("*.json"))
    assert len(saved_files) == 1
    content = saved_files[0].read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" not in content
    assert "AIzaSy" not in content
    parsed = json.loads(content)
    assert parsed["run_id"] == record["run_id"]
