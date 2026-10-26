from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, int]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) across multiple ranked lists.
    rankings: list of ranked lists where each item is (item_id, rank) with 1-indexed rank.
    k: smoothing constant (default: 60).
    Returns list of (item_id, fused_score) sorted descending by fused_score.
    """
    scores: Dict[str, float] = {}

    for ranked_list in rankings:
        for item_id, rank in ranked_list:
            if item_id not in scores:
                scores[item_id] = 0.0
            scores[item_id] += 1.0 / (k + rank)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items
