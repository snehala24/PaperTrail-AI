import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Offline flags for fast local evaluation without HF hub lookups
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from backend.app.schemas import (
    BenchmarkQuestion,
    GoldEvidence,
    Chunk,
    ChunkKind,
)
from backend.app.core.storage import storage
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.rerank import CrossEncoderReranker
from backend.app.eval.metrics import compute_retrieval_metrics, bootstrap_ci_95


def get_git_commit() -> str:
    """Retrieve current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_benchmark_data(eval_dir: Path) -> tuple[List[BenchmarkQuestion], Dict[str, List[GoldEvidence]]]:
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
    parses available PDFs in data/ (e.g. data/2610.32878v1.pdf) into storage.
    Does NOT inject any planted or synthetic benchmark passages.
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


def run_retrieval_eval() -> Dict[str, Any]:
    """
    Execute evaluation across the 4 configurations and record timestamped run log.
    - Evaluates retrieval metrics ONLY on answerable questions with gold evidence.
    - Reserves unanswerable questions for Day 3 honest abstention.
    - Calculates 95% bootstrap confidence intervals for Recall@5 and MRR (1000 resamples).
    - Emits statistical power warning when n < 30.
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

    # Partition into answerable and unanswerable questions
    answerable_questions: List[BenchmarkQuestion] = [
        q for q in questions
        if q.question_type != "unanswerable" and len(gold_by_q.get(q.question_id, [])) > 0
    ]
    unanswerable_questions: List[BenchmarkQuestion] = [
        q for q in questions if q not in answerable_questions
    ]

    configs = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]
    results_by_config: Dict[str, Dict[str, float]] = {c: {} for c in configs}
    ci_by_config: Dict[str, Dict[str, Tuple[float, float]]] = {c: {} for c in configs}
    breakdown_by_type: Dict[str, Dict[str, Dict[str, float]]] = {}

    k_list = [1, 3, 5, 10]
    all_q_metrics: Dict[str, List[Dict[str, float]]] = {c: [] for c in configs}
    q_type_metrics: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for q in answerable_questions:
        q_type = q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
        gold_items = gold_by_q.get(q.question_id, [])

        # 1. BM25 Search
        bm25_raw = bm25.search(q.question, top_k=20)
        bm25_chunks = [c for c, _, _ in bm25_raw if c.kind != ChunkKind.REFERENCES]

        # 2. Dense Search
        dense_raw = dense.search(q.question, top_k=20)
        dense_chunks = [c for c, _, _ in dense_raw if c.kind != ChunkKind.REFERENCES]

        # 3. Hybrid RRF Search
        bm25_ranks = [(c.chunk_id, r) for c, _, r in bm25_raw if c.kind != ChunkKind.REFERENCES]
        dense_ranks = [(c.chunk_id, r) for c, _, r in dense_raw if c.kind != ChunkKind.REFERENCES]
        fused = reciprocal_rank_fusion([bm25_ranks, dense_ranks], k=60)
        rrf_chunks = [chunk_map[cid] for cid, _ in fused if cid in chunk_map][:20]

        # 4. Hybrid + Rerank Search
        reranked = reranker.rerank(q.question, rrf_chunks, top_k=10)
        rerank_chunks = [c for c, _, _ in reranked]

        runs = {
            "bm25": bm25_chunks,
            "dense": dense_chunks,
            "hybrid_rrf": rrf_chunks,
            "hybrid_rerank": rerank_chunks,
        }

        for cfg_name, retrieved in runs.items():
            m = compute_retrieval_metrics(retrieved, gold_items, k_list=k_list)
            all_q_metrics[cfg_name].append(m)

            q_type_metrics.setdefault(q_type, {}).setdefault(cfg_name, []).append(m)

    # Aggregate means
    def avg_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metric_list:
            return {}
        keys = metric_list[0].keys()
        return {k: round(sum(m[k] for m in metric_list) / len(metric_list), 4) for k in keys}

    for cfg in configs:
        results_by_config[cfg] = avg_metrics(all_q_metrics[cfg])
        rec5_vals = [m.get("recall@5", 0.0) for m in all_q_metrics[cfg]]
        mrr_vals = [m.get("mrr", 0.0) for m in all_q_metrics[cfg]]
        ci_by_config[cfg] = {
            "recall@5_ci95": bootstrap_ci_95(rec5_vals, n_resamples=1000),
            "mrr_ci95": bootstrap_ci_95(mrr_vals, n_resamples=1000),
        }

    for q_type, cfg_dict in q_type_metrics.items():
        breakdown_by_type[q_type] = {}
        for cfg, m_list in cfg_dict.items():
            breakdown_by_type[q_type][cfg] = avg_metrics(m_list)

    n_eval = len(answerable_questions)
    warning_msg = None
    if n_eval < 30:
        warning_msg = f"Sample size (n={n_eval}) too small for statistical significance (n < 30)."

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_record: Dict[str, Any] = {
        "run_id": f"run_{timestamp_str}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "benchmark_provenance": "ai_draft",
        "models": {
            "embedding_model": dense.model_name,
            "reranker_model": reranker.model_name,
        },
        "chunker_settings": {
            "min_chars": 800,
            "max_chars": 1500,
            "max_table_chars": 2500,
        },
        "total_questions": len(questions),
        "answerable_questions_evaluated": n_eval,
        "unanswerable_questions_reserved": len(unanswerable_questions),
        "statistical_warning": warning_msg,
        "total_corpus_chunks": len(chunks),
        "overall_metrics_by_config": results_by_config,
        "confidence_intervals_95": ci_by_config,
        "breakdown_by_question_type": breakdown_by_type,
    }

    # Save to eval/runs/<timestamp>.json
    output_path = runs_dir / f"{run_record['run_id']}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    return run_record


def main():
    print("=" * 70)
    print("Running PaperTrail AI Retrieval Benchmark Runner")
    print("=" * 70)
    record = run_retrieval_eval()
    print(f"\nRun ID: {record['run_id']}")
    print(f"Commit: {record['git_commit']}")
    print(f"Total Questions: {record['total_questions']}")
    print(f"Answerable Evaluated: {record['answerable_questions_evaluated']}")
    print(f"Unanswerable Reserved (Day 3 Abstention): {record['unanswerable_questions_reserved']}")
    print(f"Corpus Chunks: {record['total_corpus_chunks']}")

    if record.get("statistical_warning"):
        print(f"\n>>> WARNING: {record['statistical_warning']} <<<")

    print("\n--- Overall Metrics by Configuration ---")
    header = f"{'Config':<16} | {'Recall@1':<9} | {'Recall@5 (95% CI)':<22} | {'MRR (95% CI)':<20} | {'nDCG@5':<8}"
    print(header)
    print("-" * len(header))
    for cfg, m in record["overall_metrics_by_config"].items():
        ci = record["confidence_intervals_95"].get(cfg, {})
        rec5_str = f"{m.get('recall@5', 0.0):.4f} {ci.get('recall@5_ci95', (0,0))}"
        mrr_str = f"{m.get('mrr', 0.0):.4f} {ci.get('mrr_ci95', (0,0))}"
        print(f"{cfg:<16} | {m.get('recall@1', 0.0):<9.4f} | {rec5_str:<22} | {mrr_str:<20} | {m.get('ndcg@5', 0.0):<8.4f}")

    print("\n--- Breakdown by Question Type (nDCG@5) ---")
    for q_type, cfg_data in record["breakdown_by_question_type"].items():
        print(f"Category: {q_type.upper()}")
        for cfg, m in cfg_data.items():
            print(f"  {cfg:<14}: nDCG@5 = {m.get('ndcg@5', 0.0):.4f}, Recall@5 = {m.get('recall@5', 0.0):.4f}")


if __name__ == "__main__":
    main()
