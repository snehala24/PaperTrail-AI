import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from backend.app.schemas import Chunk


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 lexical search index using rank-bm25."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: List[Chunk]) -> None:
        """Build the BM25 index from a list of chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            return

        tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float, int]]:
        """
        Search for query tokens.
        Returns list of (Chunk, score, rank) where rank is 1-indexed.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: List[Tuple[Chunk, float, int]] = []
        rank = 1
        for idx, score in scored_indices[:top_k]:
            if score > 0.0:  # Only include relevant matches
                results.append((self.chunks[idx], float(score), rank))
                rank += 1

        return results
