"""
Repeatable, offline benchmark evaluation for apparent-conflict questions (Day 3).

Evaluates the five apparent_conflict benchmark questions from eval/questions.jsonl:
- Q-005 (BitStack vs KFAC-SVD trade-off under quantization; p.23)
- Q-006 (KFAC-SVD vs SVD-LLM under uniform vs LEMS allocation; p.7)
- Q-015 (GPU used: H200 for main experiments vs A100 for inference speed; p.6, p.26)
- Q-017 (KL divergence vs MSE calibration metric; p.8)
- Q-030 (Cross-paper: LEMS claim of GFWSVD degradation vs GFWSVD paper; pending second paper)

Evaluates three orthogonal dimensions:
1. Retrieval Success: Are the required gold chunks retrieved in top-k?
2. Citation Verification Success: Do all claims pass the 3-layer verification contract?
3. Conflict Detection Correctness: Does the system identify conflicting chunks,
   flag source disagreement, and distinguish genuine contradiction from differing
   experimental conditions/metrics?

ZERO LIVE API CALLS: Fully offline and deterministic using MockLLMProvider.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce offline mode for HF models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from backend.app.citations.verifier import CitationVerifier
from backend.app.claims.service import AnswerService
import backend.app.core.storage as storage_mod
from backend.app.core.storage import DocumentStorage
from backend.app.eval.metrics import is_gold_match
from backend.app.llm.base import LLMMessage, LLMProvider, LLMResponse
from backend.app.retrieval.hybrid import ensure_retriever_indexed, retriever


def ensure_eval_storage():
    """Ensure evaluation corpus exists in current storage (populating isolated test storage if needed)."""
    current_storage = storage_mod.storage
    chunks = current_storage.get_all_chunks()
    if not chunks:
        repo_root = Path(__file__).resolve().parents[3]
        real_store = DocumentStorage(base_dir=repo_root / "data")
        real_chunks = real_store.get_all_chunks()
        real_docs = real_store.list_documents()
        for doc in real_docs:
            doc_chunks = [c for c in real_chunks if c.doc_id == doc.doc_id]
            current_storage.save_document(doc, doc_chunks, force_overwrite=True)
from backend.app.schemas import (
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    BenchmarkQuestion,
    BenchmarkQuestionType,
    BenchmarkStatus,
    ClaimVerificationStatus,
    EvidenceRelation,
    GoldEvidence,
    ScoredChunk,
)

logger = logging.getLogger(__name__)


def run_sync(coro):
    """Drive a coroutine to completion synchronously without requiring an OS socketpair."""
    if not hasattr(coro, "send"):
        return coro
    try:
        val = None
        while True:
            val = coro.send(val)
            if hasattr(val, "__await__"):
                sub = val.__await__()
                try:
                    sub_val = None
                    while True:
                        sub_val = sub.send(sub_val)
                except StopIteration as s:
                    val = s.value
            elif hasattr(val, "send"):
                val = run_sync(val)
    except StopIteration as e:
        return e.value


def get_git_commit() -> str:
    """Retrieve current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM provider for generator and verifier."""

    def __init__(self, responses: Optional[List[Union[str, LLMResponse]]] = None, default_response: str = "{}"):
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.call_history: List[List[LLMMessage]] = []

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_history.append(messages)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, LLMResponse):
                return item
            content = item
        else:
            content = self.default_response
        return LLMResponse(
            content=content,
            model="mock-gemini-2.5-flash",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


def load_conflict_benchmark_data(eval_dir: Path) -> Tuple[List[BenchmarkQuestion], Dict[str, List[GoldEvidence]]]:
    """Load benchmark questions and gold evidence filtered for apparent_conflict question type."""
    questions: List[BenchmarkQuestion] = []
    questions_file = eval_dir / "questions.jsonl"
    if questions_file.exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = BenchmarkQuestion(**json.loads(line))
                    if q.question_type == BenchmarkQuestionType.APPARENT_CONFLICT or str(q.question_type) == "apparent_conflict":
                        questions.append(q)

    gold_by_q: Dict[str, List[GoldEvidence]] = {}
    gold_file = eval_dir / "gold_evidence.jsonl"
    if gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    g = GoldEvidence(**json.loads(line))
                    gold_by_q.setdefault(g.question_id, []).append(g)

    return questions, gold_by_q


def build_deterministic_mock_for_question(
    question_id: str,
) -> Tuple[Optional[List[str]], Dict[str, Any]]:
    """
    Build deterministic generator and verifier mock responses for known apparent_conflict questions.
    All claims and quotes are strictly extracted from verified chunks in doc_c67573ef50cc.
    Returns (mock_responses_list, metadata_dict).
    """
    verif_ok = json.dumps({"supported": True, "rationale": "Exact quote and directly entailed by chunk text."})

    if question_id == "Q-005":
        # Contradiction / incompatibility: BitStack sign mask separation contradicts quantizability
        gen = json.dumps({
            "answer": (
                "KFAC-SVD achieves a better memory-accuracy trade-off than BitStack because "
                "separating out the binary mask makes combining it with quantization ineffective."
            ),
            "claims": [{
                "claim_id": "c1",
                "text": "separating out the binary mask makes combining it with quantization ineffective",
                "chunk_ids": ["doc_c67573ef50cc_p23_c3"],
                "quote": "separating out the binary mask makes combining it with quantization ineffective",
            }],
            "abstained": False,
            "abstain_reason": None,
            "conflicts": [{
                "chunk_id_a": "doc_c67573ef50cc_p23_c3",
                "chunk_id_b": "doc_c67573ef50cc_p24_c1",
            }],
        })
        return [gen, verif_ok], {
            "conflict_category": "real_contradiction",
            "resolution_reason": "Methodological incompatibility: BitStack binary sign masks cannot be compressed with standard quantization kernels.",
            "expected_conflicting_chunks": ["doc_c67573ef50cc_p23_c3", "doc_c67573ef50cc_p24_c1"],
        }

    elif question_id == "Q-006":
        # Differing rank allocation conditions: uniform allocation vs global LEMS optimization
        gen = json.dumps({
            "answer": (
                "KFAC-SVD achieves approximately 15% lower perplexity than SVD-LLM under uniform allocation, "
                "but when combined with LEMS, perplexities become tightly matched between the two methods."
            ),
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "KFAC-SVD achieves approximately 15% lower perplexity than SVD-LLM under uniform allocation",
                    "chunk_ids": ["doc_c67573ef50cc_p7_c4"],
                    "quote": "KFAC-SVD achieves approximately 15% lower perplexity than SVD-LLM under uniform allocation",
                },
                {
                    "claim_id": "c2",
                    "text": "When combined with LEMS, perplexities become tightly matched between the two methods",
                    "chunk_ids": ["doc_c67573ef50cc_p7_c4"],
                    "quote": "When combined with LEMS, perplexities become tightly matched between the two methods",
                },
            ],
            "abstained": False,
            "abstain_reason": None,
            "conflicts": [{
                "chunk_id_a": "doc_c67573ef50cc_p7_c4",
                "chunk_id_b": "doc_c67573ef50cc_p6_c5",
            }],
        })
        return [gen, verif_ok, verif_ok], {
            "conflict_category": "apparent_conflict_differing_conditions",
            "resolution_reason": "Differing rank allocation regime: KFAC-SVD outperforms under uniform allocation, but performance tightly converges under global LEMS optimization.",
            "expected_conflicting_chunks": ["doc_c67573ef50cc_p7_c4", "doc_c67573ef50cc_p6_c5"],
        }

    elif question_id == "Q-015":
        # Differing hardware conditions: H200 for main experiments vs A100 for inference speed
        gen = json.dumps({
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
        return [gen, verif_ok, verif_ok], {
            "conflict_category": "apparent_conflict_differing_conditions",
            "resolution_reason": "Differing experimental hardware: NVIDIA H200 GPU used for main experiments (p.6), whereas NVIDIA A100 GPU was used for wall-clock inference speedup benchmarking (p.26).",
            "expected_conflicting_chunks": ["doc_c67573ef50cc_p6_c2", "doc_c67573ef50cc_p26_c7"],
        }

    elif question_id == "Q-017":
        # Differing metrics: zero-shot accuracy vs perplexity
        gen = json.dumps({
            "answer": (
                "KL divergence consistently outperforms MSE in zero shot accuracy, while for perplexity, "
                "MSE performs slightly better than KL, but with small margins."
            ),
            "claims": [
                {
                    "claim_id": "c1",
                    "text": "KL divergence consistently outperforms MSE in zero shot accuracy",
                    "chunk_ids": ["doc_c67573ef50cc_p8_c6"],
                    "quote": "KL divergence consistently outperforms MSE in zero shot accuracy",
                },
                {
                    "claim_id": "c2",
                    "text": "For perplexity, MSE performs slightly better than KL, but with small margins.",
                    "chunk_ids": ["doc_c67573ef50cc_p8_c6"],
                    "quote": "For perplexity, MSE performs slightly better than KL, but with small margins.",
                },
            ],
            "abstained": False,
            "abstain_reason": None,
            "conflicts": [{
                "chunk_id_a": "doc_c67573ef50cc_p8_c6",
                "chunk_id_b": "doc_c67573ef50cc_p18_c1",
            }],
        })
        return [gen, verif_ok, verif_ok], {
            "conflict_category": "apparent_conflict_differing_conditions",
            "resolution_reason": "Differing evaluation metrics: KL divergence outperforms MSE on zero-shot accuracy (+1.1 p.p.), while MSE performs slightly better on perplexity (p.8).",
            "expected_conflicting_chunks": ["doc_c67573ef50cc_p8_c6", "doc_c67573ef50cc_p18_c1"],
        }

    return None, {}


def run_conflicts_eval(
    eval_dir: Optional[Path] = None,
    runs_dir: Optional[Path] = None,
    save_artifact: bool = True,
) -> Dict[str, Any]:
    """
    Execute repeatable offline apparent-conflict benchmark evaluation.
    - Loads all 5 apparent_conflict questions.
    - Flags Q-030 (pending second paper) as BLOCKED_PENDING_SECOND_PAPER.
    - Evaluates retrieval, 3-layer verification, and conflict detection on evaluable questions.
    - Generates sanitized benchmark artifact.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if eval_dir is None:
        eval_dir = repo_root / "eval"
    if runs_dir is None:
        runs_dir = eval_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ensure_eval_storage()
    ensure_retriever_indexed(force=True)

    questions, gold_by_q = load_conflict_benchmark_data(eval_dir)

    results_per_question: List[Dict[str, Any]] = []

    retrieval_success_count = 0
    verification_success_count = 0
    conflict_detection_success_count = 0
    overall_grounded_success_count = 0
    evaluated_count = 0
    blocked_count = 0

    for q in sorted(questions, key=lambda x: x.question_id):
        qid = q.question_id
        golds = gold_by_q.get(qid, [])

        # Check multi-paper prerequisite
        is_blocked = (
            q.status == BenchmarkStatus.PENDING_SECOND_PAPER
            or "doc_gfwsvd_placeholder" in q.target_doc_ids
            or any("TODO" in g.doc_sha256 or (g.answer_span and g.answer_span.startswith("TODO")) for g in golds)
        )

        if is_blocked:
            blocked_count += 1
            results_per_question.append({
                "question_id": qid,
                "question": q.question,
                "question_type": "apparent_conflict",
                "status": "BLOCKED_PENDING_SECOND_PAPER",
                "evaluable": False,
                "block_reason": "Cross-paper question requires second paper (GFWSVD / TODO_GFWSVD_SHA256) which is not yet ingested into corpus.",
                "target_doc_ids": q.target_doc_ids,
                "retrieval_success": None,
                "citation_verification_success": None,
                "conflict_detection_success": None,
                "overall_success": None,
            })
            continue

        evaluated_count += 1

        # -------------------------------------------------------------
        # 1. Retrieval Evaluation
        # -------------------------------------------------------------
        active_golds = [
            g for g in golds
            if not ("TODO" in g.doc_sha256 or (g.answer_span and g.answer_span.startswith("TODO")))
        ]

        retrieved: List[ScoredChunk] = retriever.search(
            query=q.question,
            top_k=5,
            doc_ids=q.target_doc_ids,
            use_reranker=True,
        )

        matched_golds = set()
        for rank, sc in enumerate(retrieved, start=1):
            for g in active_golds:
                if is_gold_match(sc, g):
                    matched_golds.add(g.gold_id)

        retrieval_recall = len(matched_golds) / len(active_golds) if active_golds else 0.0
        retrieval_success = (retrieval_recall == 1.0)
        if retrieval_success:
            retrieval_success_count += 1

        # -------------------------------------------------------------
        # 2. Generation & Citation Verification Evaluation
        # -------------------------------------------------------------
        mock_responses, meta = build_deterministic_mock_for_question(qid)
        if not mock_responses:
            results_per_question.append({
                "question_id": qid,
                "question": q.question,
                "status": "ERROR_NO_MOCK",
                "evaluable": True,
                "retrieval_success": retrieval_success,
                "citation_verification_success": False,
                "conflict_detection_success": False,
                "overall_success": False,
            })
            continue

        mock_llm = MockLLMProvider(responses=mock_responses)
        verifier = CitationVerifier(llm_provider=mock_llm)
        service = AnswerService(retriever=retriever, llm_provider=mock_llm, verifier=verifier)

        req = AnswerRequest(
            question=q.question,
            top_k=5,
            doc_ids=q.target_doc_ids,
        )
        answer_resp: AnswerResponse = run_sync(service.answer(req))

        all_checks = answer_resp.citation_checks
        all_checks_verified = (
            len(all_checks) > 0
            and all(c.status == ClaimVerificationStatus.VERIFIED for c in all_checks)
        )
        verification_success = (
            answer_resp.status == AnswerStatus.ANSWERED
            and all_checks_verified
        )
        if verification_success:
            verification_success_count += 1

        # -------------------------------------------------------------
        # 3. Conflict Detection Evaluation
        # -------------------------------------------------------------
        answer_text = answer_resp.answer
        flags_source_disagreement = "sources disagree" in answer_text.lower()
        expected_chunks = meta.get("expected_conflicting_chunks", [])
        identifies_chunks = all(cid in answer_text for cid in expected_chunks)

        conflict_detection_success = flags_source_disagreement and identifies_chunks
        if conflict_detection_success:
            conflict_detection_success_count += 1

        overall_success = retrieval_success and verification_success and conflict_detection_success
        if overall_success:
            overall_grounded_success_count += 1

        results_per_question.append({
            "question_id": qid,
            "question": q.question,
            "question_type": "apparent_conflict",
            "conflict_category": meta.get("conflict_category"),
            "resolution_reason": meta.get("resolution_reason"),
            "status": "EVALUATED_GROUNDED" if overall_success else "EVALUATED_FAILED",
            "evaluable": True,
            "retrieval": {
                "success": retrieval_success,
                "recall@5": round(retrieval_recall, 4),
                "active_gold_count": len(active_golds),
                "matched_gold_count": len(matched_golds),
                "top_retrieved_chunks": [c.chunk_id for c in retrieved[:3]],
            },
            "citation_verification": {
                "success": verification_success,
                "answer_status": answer_resp.status.value,
                "citation_checks_count": len(all_checks),
                "all_citations_verified": all_checks_verified,
            },
            "conflict_detection": {
                "success": conflict_detection_success,
                "flags_source_disagreement": flags_source_disagreement,
                "identifies_conflicting_chunks": identifies_chunks,
                "expected_chunks": expected_chunks,
            },
            "overall_success": overall_success,
            "final_answer": answer_resp.answer,
        })

    # Summary metrics with explicit denominators
    retrieval_rate = round(retrieval_success_count / evaluated_count, 4) if evaluated_count else 0.0
    verification_rate = round(verification_success_count / evaluated_count, 4) if evaluated_count else 0.0
    conflict_rate = round(conflict_detection_success_count / evaluated_count, 4) if evaluated_count else 0.0
    overall_rate = round(overall_grounded_success_count / evaluated_count, 4) if evaluated_count else 0.0

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_record: Dict[str, Any] = {
        "run_id": f"run_{timestamp_str}_conflicts",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "evaluation_type": "apparent_conflict_grounding_and_verification",
        "summary": {
            "total_conflict_questions": len(questions),
            "evaluated_questions": evaluated_count,
            "blocked_pending_second_paper": blocked_count,
            "retrieval_success_count": retrieval_success_count,
            "retrieval_success_rate": retrieval_rate,
            "citation_verification_success_count": verification_success_count,
            "citation_verification_success_rate": verification_rate,
            "conflict_detection_success_count": conflict_detection_success_count,
            "conflict_detection_success_rate": conflict_rate,
            "overall_grounded_success_count": overall_grounded_success_count,
            "overall_grounded_success_rate": overall_rate,
        },
        "per_question_results": results_per_question,
    }

    if save_artifact:
        output_path = runs_dir / f"{run_record['run_id']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)

    return run_record


def print_conflict_results_table(record: Dict[str, Any]):
    """Print formatted summary table of apparent-conflict evaluation."""
    s = record["summary"]
    print("\n" + "=" * 90)
    print("  PAPERTRAIL AI: APPARENT-CONFLICT BENCHMARK EVALUATION (DAY 3)")
    print("=" * 90)
    print(f"  Total Apparent-Conflict Questions : {s['total_conflict_questions']}")
    print(f"  Evaluated (Single-Paper Corpus)  : {s['evaluated_questions']}")
    print(f"  Blocked (Pending Second Paper)   : {s['blocked_pending_second_paper']} (Q-030)")
    print(f"  Retrieval Success Rate           : {s['retrieval_success_count']}/{s['evaluated_questions']} ({s['retrieval_success_rate'] * 100:.1f}%)")
    print(f"  Citation Verification Rate       : {s['citation_verification_success_count']}/{s['evaluated_questions']} ({s['citation_verification_success_rate'] * 100:.1f}%)")
    print(f"  Conflict Detection Correctness   : {s['conflict_detection_success_count']}/{s['evaluated_questions']} ({s['conflict_detection_success_rate'] * 100:.1f}%)")
    print(f"  Overall Grounded Success Rate    : {s['overall_grounded_success_count']}/{s['evaluated_questions']} ({s['overall_grounded_success_rate'] * 100:.1f}%)")
    print("=" * 90)

    hdr = f"{'Q-ID':<7} | {'Category':<32} | {'Retrieval':<10} | {'Verification':<13} | {'Conflict Flag':<14} | {'Status':<16}"
    print(hdr)
    print("-" * len(hdr))

    for item in record["per_question_results"]:
        qid = item["question_id"]
        cat = item.get("conflict_category") or "cross_paper (pending)"
        if not item.get("evaluable"):
            print(f"{qid:<7} | {cat:<32} | {'N/A':<10} | {'N/A':<13} | {'N/A':<14} | {'BLOCKED':<16}")
            continue

        ret_str = "PASS" if item["retrieval"]["success"] else "FAIL"
        ver_str = "PASS" if item["citation_verification"]["success"] else "FAIL"
        cnf_str = "PASS" if item["conflict_detection"]["success"] else "FAIL"
        sta_str = "GROUNDED" if item["overall_success"] else "FAILED"
        print(f"{qid:<7} | {cat:<32} | {ret_str:<10} | {ver_str:<13} | {cnf_str:<14} | {sta_str:<16}")

    print("=" * 90)


def main():
    print("Running PaperTrail AI Apparent-Conflict Benchmark...")
    record = run_conflicts_eval()
    print_conflict_results_table(record)
    print(f"\nSaved benchmark run record to eval/runs/{record['run_id']}.json")


if __name__ == "__main__":
    main()
