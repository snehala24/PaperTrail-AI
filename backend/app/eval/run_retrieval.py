import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Offline flags for fast local evaluation without HF hub lookups
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from backend.app.schemas import (
    BenchmarkQuestion,
    BenchmarkStatus,
    GoldEvidence,
    Chunk,
    ChunkKind,
)
from backend.app.core.storage import storage
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.rerank import CrossEncoderReranker
from backend.app.retrieval.hybrid import select_allowed_candidates
from backend.app.eval.metrics import compute_retrieval_metrics, bootstrap_ci_95, is_gold_match

# Configurable candidate-pool size passed from fusion to cross-encoder (default 30)
RERANK_POOL: int = int(os.environ.get("RERANK_POOL", "30"))


def get_git_commit() -> str:
    """Retrieve current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_benchmark_data(eval_dir: Path) -> Tuple[List[BenchmarkQuestion], Dict[str, List[GoldEvidence]]]:
    """Load benchmark questions and group gold evidence by question_id."""
    questions: List[BenchmarkQuestion] = []
    questions_file = eval_dir / "questions.jsonl"
    if questions_file.exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(BenchmarkQuestion(**json.loads(line)))

    gold_by_q: Dict[str, List[GoldEvidence]] = {}
    gold_file = eval_dir / "gold_evidence.jsonl"
    if gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    g = GoldEvidence(**json.loads(line))
                    gold_by_q.setdefault(g.question_id, []).append(g)

    return questions, gold_by_q


def ensure_eval_corpus(gold_by_q: Dict[str, List[GoldEvidence]]) -> List[Chunk]:
    """
    Ensure an indexed corpus exists from stored documents.
    Loads real documents and chunks from storage. If no chunks exist in storage,
    parses available PDFs in data/ into storage.
    """
    chunks = storage.get_all_chunks()
    if chunks:
        return chunks

    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "data" / "2610.32878v1.pdf"
    if pdf_path.exists():
        from backend.app.ingestion.parser import parse_pdf_bytes

        doc, parsed_chunks = parse_pdf_bytes(pdf_path.read_bytes(), pdf_path.name)
        storage.save_document(doc, parsed_chunks, force_overwrite=True)
        return parsed_chunks

    return []


def get_first_relevant_rank(chunks: List[Chunk], gold_items: List[GoldEvidence]) -> Optional[int]:
    """Return 1-indexed rank of the first chunk that matches any active gold item."""
    for rank, chunk in enumerate(chunks, start=1):
        if any(is_gold_match(chunk, g) for g in gold_items):
            return rank
    return None


def aggregate_metrics_for_subset(
    questions: List[BenchmarkQuestion],
    per_q_eval: Dict[str, Dict[str, Any]],
    configs: List[str] = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"],
) -> Dict[str, Any]:
    """Aggregate metrics and compute 95% bootstrap CIs for a filtered subset of questions."""
    eval_qids = [q.question_id for q in questions if q.question_id in per_q_eval]
    unanswerable_count = len([q for q in questions if q.question_type == "unanswerable"])

    all_q_metrics: Dict[str, List[Dict[str, float]]] = {c: [] for c in configs}
    q_type_metrics: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for qid in eval_qids:
        entry = per_q_eval[qid]
        q_type = entry["q_type"]
        for cfg in configs:
            m = entry["metrics"][cfg]
            all_q_metrics[cfg].append(m)
            q_type_metrics.setdefault(q_type, {}).setdefault(cfg, []).append(m)

    def avg_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metric_list:
            return {}
        keys = metric_list[0].keys()
        return {k: round(sum(m[k] for m in metric_list) / len(metric_list), 4) for k in keys}

    results_by_config: Dict[str, Dict[str, float]] = {}
    ci_by_config: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for cfg in configs:
        results_by_config[cfg] = avg_metrics(all_q_metrics[cfg])
        rec5_vals = [m.get("recall@5", 0.0) for m in all_q_metrics[cfg]]
        mrr_vals = [m.get("mrr", 0.0) for m in all_q_metrics[cfg]]
        ci_by_config[cfg] = {
            "recall@5_ci95": bootstrap_ci_95(rec5_vals, n_resamples=1000),
            "mrr_ci95": bootstrap_ci_95(mrr_vals, n_resamples=1000),
        }

    breakdown_by_type: Dict[str, Dict[str, Dict[str, float]]] = {}
    for q_type, cfg_dict in q_type_metrics.items():
        breakdown_by_type[q_type] = {}
        for cfg, m_list in cfg_dict.items():
            breakdown_by_type[q_type][cfg] = avg_metrics(m_list)

    n_eval = len(eval_qids)
    warning_msg = None
    if n_eval < 30:
        warning_msg = f"Sample size (n={n_eval}) too small for statistical significance (n < 30)."

    return {
        "n_total": len(questions),
        "n_answerable_evaluated": n_eval,
        "n_unanswerable_reserved": unanswerable_count,
        "statistical_warning": warning_msg,
        "overall_metrics_by_config": results_by_config,
        "confidence_intervals_95": ci_by_config,
        "breakdown_by_question_type": breakdown_by_type,
    }


def build_eval_rrf_pool(
    bm25_raw: List[Tuple[Chunk, float, int]],
    dense_raw: List[Tuple[Chunk, float, int]],
    chunk_map: Dict[str, Chunk],
    pool_size: int = RERANK_POOL,
) -> Tuple[List[Chunk], List[Chunk]]:
    """
    Build fused candidate pools for evaluation from raw BM25 and dense retrieval results.
    Uses select_allowed_candidates to filter and cap each modality to pool_size candidates.
    Returns:
        (fused_chunks_all, fused_pool_chunks)
    """
    bm25_candidates = select_allowed_candidates(bm25_raw, max_candidates=pool_size)
    dense_candidates = select_allowed_candidates(dense_raw, max_candidates=pool_size)
    bm25_ranks_for_fusion = [(c.chunk_id, r) for c, _, r in bm25_candidates]
    dense_ranks_for_fusion = [(c.chunk_id, r) for c, _, r in dense_candidates]
    fused = reciprocal_rank_fusion([bm25_ranks_for_fusion, dense_ranks_for_fusion], k=60)
    fused_chunks_all = [chunk_map[cid] for cid, _ in fused if cid in chunk_map]
    fused_pool_chunks = fused_chunks_all[:pool_size]
    return fused_chunks_all, fused_pool_chunks


def run_retrieval_eval() -> Dict[str, Any]:
    """
    Execute single-pass retrieval across configurations and record timestamped run log.
    - BM25 and Dense each contribute top 50 to allow failure tracking up to rank 50.
    - Candidate pool passed from fusion to cross-encoder is configurable (RERANK_POOL, default 30).
    - Reranks the whole fused pool (top RERANK_POOL).
    - Excludes cross-paper stubs (Q-029, Q-030) pending GFWSVD ingestion.
    - Generates detailed per-question first-relevant-rank and Top-1 chunk analysis.
    """
    repo_root = Path(__file__).resolve().parents[3]
    eval_dir = repo_root / "eval"
    runs_dir = eval_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    questions, gold_by_q = load_benchmark_data(eval_dir)
    chunks = ensure_eval_corpus(gold_by_q)

    # Initialize retriever components
    bm25 = BM25Index()
    bm25.build(chunks)

    dense = DenseIndex()
    dense.build(chunks)

    reranker = CrossEncoderReranker()
    chunk_map = {c.chunk_id: c for c in chunks}
    configs = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]
    k_list = [1, 3, 5, 10]

    # Search depth: pull up to max(RERANK_POOL * 2, 50) for failure tracking and 2x candidate pool buffer
    search_depth = max(RERANK_POOL * 2, 50)

    per_q_eval: Dict[str, Dict[str, Any]] = {}
    failure_analysis: List[Dict[str, Any]] = []

    for q in questions:
        # Exclude unanswerables and cross-paper pending items from evaluation
        if q.question_type == "unanswerable" or q.status == BenchmarkStatus.PENDING_SECOND_PAPER:
            continue
        if q.question_id in ("Q-029", "Q-030"):
            continue

        g_items = gold_by_q.get(q.question_id, [])
        active_golds = [
            g for g in g_items
            if not ("TODO" in g.doc_sha256 or (g.answer_span and g.answer_span.startswith("TODO")))
        ]
        if not active_golds:
            continue

        q_type = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)

        # 1. BM25 Search (top 50)
        bm25_raw = bm25.search(q.question, top_k=search_depth)
        bm25_chunks = [c for c, _, _ in bm25_raw if c.kind != ChunkKind.REFERENCES]

        # 2. Dense Search (top 50)
        dense_raw = dense.search(q.question, top_k=search_depth)
        dense_chunks = [c for c, _, _ in dense_raw if c.kind != ChunkKind.REFERENCES]

        # 3. Hybrid RRF Search (fuse BM25 top 30 + Dense top 30 allowed candidates)
        fused_chunks_all, fused_pool_chunks = build_eval_rrf_pool(
            bm25_raw=bm25_raw,
            dense_raw=dense_raw,
            chunk_map=chunk_map,
            pool_size=RERANK_POOL,
        )

        # 4. Cross-Encoder Rerank (rerank whole fused candidate pool of size RERANK_POOL)
        reranked = reranker.rerank(q.question, fused_pool_chunks, top_k=RERANK_POOL)
        rerank_chunks = [c for c, _, _ in reranked]

        # Compute standard retrieval metrics
        runs = {
            "bm25": bm25_chunks,
            "dense": dense_chunks,
            "hybrid_rrf": fused_pool_chunks,
            "hybrid_rerank": rerank_chunks,
        }

        q_cfg_metrics: Dict[str, Dict[str, float]] = {}
        for cfg_name, retrieved in runs.items():
            q_cfg_metrics[cfg_name] = compute_retrieval_metrics(retrieved, active_golds, k_list=k_list)

        per_q_eval[q.question_id] = {
            "q_type": q_type,
            "metrics": q_cfg_metrics,
        }

        # Track first relevant rank and Top-1 chunk for failure analysis
        bm25_rel_rank = get_first_relevant_rank(bm25_chunks, active_golds)
        dense_rel_rank = get_first_relevant_rank(dense_chunks, active_golds)
        rrf_rel_rank = get_first_relevant_rank(fused_chunks_all, active_golds)
        rerank_rel_rank = get_first_relevant_rank(rerank_chunks, active_golds)

        def get_top1_info(chunk_list: List[Chunk]) -> Dict[str, Any]:
            if not chunk_list:
                return {"chunk_id": "none", "page": 0, "heading": "none"}
            top = chunk_list[0]
            return {
                "chunk_id": top.chunk_id,
                "page": top.page,
                "heading": top.section_heading or "No heading",
            }

        q_analysis = {
            "question_id": q.question_id,
            "question": q.question,
            "question_type": q_type,
            "labelled_by": q.labelled_by,
            "status": q.status.value if hasattr(q.status, "value") else str(q.status),
            "ranks": {
                "bm25": bm25_rel_rank if bm25_rel_rank is not None else ">50",
                "dense": dense_rel_rank if dense_rel_rank is not None else ">50",
                "hybrid_rrf": rrf_rel_rank if rrf_rel_rank is not None else ">50",
                "hybrid_rerank": rerank_rel_rank if rerank_rel_rank is not None else f">{RERANK_POOL}",
            },
            "top1_chunks": {
                "bm25": get_top1_info(bm25_chunks),
                "dense": get_top1_info(dense_chunks),
                "hybrid_rrf": get_top1_info(fused_pool_chunks),
                "hybrid_rerank": get_top1_info(rerank_chunks),
            },
        }
        failure_analysis.append(q_analysis)

    # Partition questions by provenance
    human_questions = [
        q for q in questions
        if (q.labelled_by != "assistant" or q.status in (BenchmarkStatus.HUMAN_DRAFT, BenchmarkStatus.HUMAN_VERIFIED, BenchmarkStatus.VERIFIED))
        and q.status != BenchmarkStatus.PENDING_SECOND_PAPER
        and q.question_id not in ("Q-029", "Q-030")
    ]
    ai_questions = [
        q for q in questions
        if q.labelled_by == "assistant" or q.status == BenchmarkStatus.AI_DRAFT
    ]

    human_eval = aggregate_metrics_for_subset(human_questions, per_q_eval, configs=configs)
    ai_eval = aggregate_metrics_for_subset(ai_questions, per_q_eval, configs=configs)
    combined_eval = aggregate_metrics_for_subset(human_questions + ai_questions, per_q_eval, configs=configs)

    # Identify Recall@5 failures (first relevant rank > 5 or not found)
    human_analysis = [qa for qa in failure_analysis if qa["question_id"].startswith("Q-01") or qa["question_id"].startswith("Q-02")]

    def is_fail(rank_val: Any) -> bool:
        if isinstance(rank_val, int):
            return rank_val > 5
        return True

    recall5_failures = {
        cfg: [qa["question_id"] for qa in human_analysis if is_fail(qa["ranks"][cfg])]
        for cfg in configs
    }

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_record: Dict[str, Any] = {
        "run_id": f"run_{timestamp_str}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "candidate_pool_size": RERANK_POOL,
        "models": {
            "embedding_model": dense.model_name,
            "reranker_model": reranker.model_name,
        },
        "chunker_settings": {
            "min_chars": 800,
            "max_chars": 1500,
            "max_table_chars": 2500,
        },
        "total_corpus_chunks": len(chunks),
        "total_questions": len(questions),
        "human_labeled": human_eval,
        "ai_draft": ai_eval,
        "combined": combined_eval,
        "human_failure_analysis": human_analysis,
        "human_recall5_failures_by_config": recall5_failures,
    }

    # Save to eval/runs/<timestamp>.json
    output_path = runs_dir / f"{run_record['run_id']}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    return run_record


def print_results_table(title: str, eval_data: Dict[str, Any]):
    print(f"\n{'=' * 75}")
    print(f"  {title.upper()}")
    print(f"  Questions Evaluated (Answerable): n = {eval_data['n_answerable_evaluated']}  |  Reserved Unanswerable: {eval_data['n_unanswerable_reserved']}")
    if eval_data.get("statistical_warning"):
        print(f"  >>> WARNING: {eval_data['statistical_warning']} <<<")
    print(f"{'=' * 75}")

    header = f"{'Config':<16} | {'Recall@1':<9} | {'Recall@5 (95% CI)':<22} | {'MRR (95% CI)':<20} | {'nDCG@5':<8}"
    print(header)
    print("-" * len(header))
    for cfg, m in eval_data["overall_metrics_by_config"].items():
        ci = eval_data["confidence_intervals_95"].get(cfg, {})
        rec5_str = f"{m.get('recall@5', 0.0):.4f} {ci.get('recall@5_ci95', (0,0))}"
        mrr_str = f"{m.get('mrr', 0.0):.4f} {ci.get('mrr_ci95', (0,0))}"
        print(f"{cfg:<16} | {m.get('recall@1', 0.0):<9.4f} | {rec5_str:<22} | {mrr_str:<20} | {m.get('ndcg@5', 0.0):<8.4f}")

    print(f"\n--- Breakdown by Question Type (nDCG@5 / Recall@5) ---")
    for q_type, cfg_data in eval_data.get("breakdown_by_question_type", {}).items():
        print(f"Category: {q_type.upper()}")
        for cfg, m in cfg_data.items():
            print(f"  {cfg:<14}: nDCG@5 = {m.get('ndcg@5', 0.0):.4f}, Recall@5 = {m.get('recall@5', 0.0):.4f}")


def print_failure_analysis_table(record: Dict[str, Any]):
    analysis: List[Dict[str, Any]] = record.get("human_failure_analysis", [])
    pool_size: int = record.get("candidate_pool_size", 30)

    print(f"\n{'=' * 110}")
    print(f"  PER-QUESTION RETRIEVAL FAILURE ANALYSIS (HUMAN-LABELLED SET, n={len(analysis)})")
    print(f"  Candidate Pool Size to Cross-Encoder: RERANK_POOL = {pool_size}")
    print(f"{'=' * 110}")

    hdr = f"{'Q-ID':<7} | {'BM25':<6} | {'Dense':<6} | {'RRF':<6} | {'Rerank':<7} | {'Top-1 Chunk (Hybrid + Rerank)':<70}"
    print(hdr)
    print("-" * len(hdr))

    for item in analysis:
        qid = item["question_id"]
        ranks = item["ranks"]
        top_rerank = item["top1_chunks"]["hybrid_rerank"]
        top_str = f"{top_rerank['chunk_id']} (p.{top_rerank['page']}: {top_rerank['heading'][:42]})"

        bm25_r = str(ranks["bm25"])
        dense_r = str(ranks["dense"])
        rrf_r = str(ranks["hybrid_rrf"])
        rerank_r = str(ranks["hybrid_rerank"])

        print(f"{qid:<7} | {bm25_r:<6} | {dense_r:<6} | {rrf_r:<6} | {rerank_r:<7} | {top_str:<70}")

    print(f"\n{'=' * 110}")
    print("  SUMMARY: QUESTIONS FAILING RECALL@5 (FIRST RELEVANT CHUNK RANK > 5)")
    print(f"{'=' * 110}")
    failures = record.get("human_recall5_failures_by_config", {})
    for cfg, qids in failures.items():
        print(f"  {cfg:<15}: {len(qids)} failures -> {qids if qids else 'None (100% Recall@5)'}")
    print(f"{'=' * 110}")


def main():
    print("=" * 75)
    print("Running PaperTrail AI Retrieval Benchmark Runner")
    print("=" * 75)
    record = run_retrieval_eval()
    print(f"\nRun ID: {record['run_id']}")
    print(f"Commit: {record['git_commit']}")
    print(f"Corpus Chunks: {record['total_corpus_chunks']}")
    print(f"Candidate Pool Size to Cross-Encoder: {record['candidate_pool_size']}")
    print(f"Total Benchmark Questions: {record['total_questions']}")

    print_results_table("1. Human-Labelled Benchmark (Annotator: snehala24)", record["human_labeled"])
    print_results_table("2. AI-Draft Benchmark (Annotator: assistant - Pipeline Validation)", record["ai_draft"])
    print_results_table("3. Combined Benchmark (Human + AI-Draft)", record["combined"])

    print_failure_analysis_table(record)


if __name__ == "__main__":
    main()
