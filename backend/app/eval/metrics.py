import math
from typing import Dict, List
from backend.app.schemas import Chunk, GoldEvidence
from backend.app.schemas.eval import is_chunk_match


def compute_retrieval_metrics(
    retrieved_chunks: List[Chunk],
    gold_items: List[GoldEvidence],
    k_list: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute Recall@k, Precision@k, MRR, and nDCG@k against gold evidence items.
    Relevance is determined via is_chunk_match:
    chunk matches if chunk.page == gold.page and gold.answer_span is in chunk.text (case-insensitive, normalized whitespace).
    """
    metrics: Dict[str, float] = {}

    if not gold_items:
        # If no gold items (e.g. unanswerable question with no evidence in corpus)
        for k in k_list:
            metrics[f"recall@{k}"] = 1.0 if not retrieved_chunks else 0.0
            metrics[f"precision@{k}"] = 1.0 if not retrieved_chunks else 0.0
            metrics[f"ndcg@{k}"] = 1.0 if not retrieved_chunks else 0.0
        metrics["mrr"] = 1.0 if not retrieved_chunks else 0.0
        return metrics

    # Identify relevance of each retrieved chunk (1 if it matches any gold item, else 0)
    relevance_flags: List[int] = []
    first_hit_rank: int | None = None

    for rank_idx, chunk in enumerate(retrieved_chunks, start=1):
        is_relevant = any(
            is_chunk_match(chunk.page, chunk.text, g.page, g.answer_span)
            for g in gold_items
        )
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
        metrics[f"precision@{k}"] = prec

        # Recall@k = how many unique gold items are covered in top k / total gold items
        matched_gold_count = 0
        for g in gold_items:
            if any(
                is_chunk_match(c.page, c.text, g.page, g.answer_span)
                for c in retrieved_chunks[:k]
            ):
                matched_gold_count += 1
        metrics[f"recall@{k}"] = matched_gold_count / num_gold if num_gold > 0 else 0.0

        # DCG@k and IDCG@k
        dcg = sum((rel / math.log2(idx + 1)) for idx, rel in enumerate(sub_flags, start=1))
        # Ideal DCG: top min(num_gold, k) are all 1
        ideal_count = min(num_gold, k)
        idcg = sum((1.0 / math.log2(idx + 1)) for idx in range(1, ideal_count + 1))
        metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0.0 else (1.0 if num_gold == 0 else 0.0)

    return metrics
