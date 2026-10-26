import math
import random
from typing import Dict, List, Optional, Tuple
from backend.app.schemas import Chunk, GoldEvidence
from backend.app.schemas.eval import is_chunk_match


def is_gold_match(
    chunk: Chunk,
    gold: GoldEvidence,
    chunk_doc_sha256: Optional[str] = None,
) -> bool:
    """Check if a retrieved chunk matches a gold evidence item."""
    if gold.gold_chunk_ids and chunk.chunk_id in gold.gold_chunk_ids:
        return True
    if chunk_doc_sha256 and gold.doc_sha256 and chunk_doc_sha256.lower() != gold.doc_sha256.lower():
        return False
    return is_chunk_match(chunk.page, chunk.text, gold.page, gold.answer_span)


def compute_retrieval_metrics(
    retrieved_chunks: List[Chunk],
    gold_items: List[GoldEvidence],
    k_list: List[int] = [1, 3, 5, 10],
    chunk_sha256_map: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Compute Recall@k, Precision@k, MRR, and nDCG@k against gold evidence items.
    Must be called only on answerable questions with gold evidence items.
    """
    if not gold_items:
        raise ValueError(
            "Cannot compute retrieval metrics for question with no gold evidence items. "
            "Unanswerable questions should be excluded from retrieval metrics."
        )

    metrics: Dict[str, float] = {}

    def chunk_matches_any_gold(chunk: Chunk) -> bool:
        doc_sha = chunk_sha256_map.get(chunk.chunk_id) if chunk_sha256_map else None
        return any(is_gold_match(chunk, g, doc_sha) for g in gold_items)

    # Identify relevance of each retrieved chunk (1 if it matches any gold item, else 0)
    relevance_flags: List[int] = []
    first_hit_rank: Optional[int] = None

    for rank_idx, chunk in enumerate(retrieved_chunks, start=1):
        is_relevant = chunk_matches_any_gold(chunk)
        flag = 1 if is_relevant else 0
        relevance_flags.append(flag)
        if flag == 1 and first_hit_rank is None:
            first_hit_rank = rank_idx

    # MRR (Mean Reciprocal Rank)
    metrics["mrr"] = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0

    # Recall@k, Precision@k, nDCG@k
    num_gold = len(gold_items)

    for k in k_list:
        sub_flags = relevance_flags[:k]

        # Precision@k = relevant retrieved in top k / k
        prec = sum(sub_flags) / k if k > 0 else 0.0
        metrics[f"precision@{k}"] = round(prec, 4)

        # Recall@k = how many unique gold items are covered in top k / total gold items
        matched_gold_count = 0
        for g in gold_items:
            if any(
                is_gold_match(
                    c,
                    g,
                    chunk_sha256_map.get(c.chunk_id) if chunk_sha256_map else None,
                )
                for c in retrieved_chunks[:k]
            ):
                matched_gold_count += 1
        metrics[f"recall@{k}"] = round(matched_gold_count / num_gold, 4) if num_gold > 0 else 0.0

        # DCG@k and IDCG@k
        dcg = sum((rel / math.log2(idx + 1)) for idx, rel in enumerate(sub_flags, start=1))
        ideal_count = min(num_gold, k)
        idcg = sum((1.0 / math.log2(idx + 1)) for idx in range(1, ideal_count + 1))
        metrics[f"ndcg@{k}"] = round(dcg / idcg, 4) if idcg > 0.0 else 0.0

    return metrics


def bootstrap_ci_95(
    values: List[float],
    n_resamples: int = 1000,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute 95% bootstrap confidence interval (percentile bootstrap) for mean of values.
    Returns (lower_bound, upper_bound).
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (round(values[0], 4), round(values[0], 4))

    rng = random.Random(random_seed)
    n = len(values)
    means: List[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lower_idx = int(0.025 * n_resamples)
    upper_idx = int(0.975 * n_resamples)
    return (round(means[lower_idx], 4), round(means[upper_idx], 4))

    return metrics
