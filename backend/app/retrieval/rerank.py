import os
from typing import List, Optional, Tuple
from backend.app.schemas import Chunk


class CrossEncoderReranker:
    """Cross-encoder reranker for fine-grained query-chunk semantic relevance."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: List[Chunk], top_k: int = 10
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Score (query, chunk_text) pairs and return top_k reranked candidates.
        Returns list of (Chunk, rerank_score, rank) with 1-indexed rank.
        """
        if not candidates:
            return []

        pairs = [[query, f"{c.section_heading or ''}\n{c.text}".strip()] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[Chunk, float, int]] = []
        for rank, (chunk, score) in enumerate(scored_candidates[:top_k], start=1):
            results.append((chunk, float(score), rank))

        return results
