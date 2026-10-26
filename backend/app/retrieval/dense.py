import os
from typing import List, Optional, Tuple
import faiss
import numpy as np
from backend.app.schemas import Chunk


class DenseIndex:
    """FAISS CPU Dense Vector Index with sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self._model = None
        self.chunks: List[Chunk] = []
        self.index = None
        self.dimension: int = 384  # Default dimension for all-MiniLM-L6-v2

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimension = self._model.get_embedding_dimension()
            else:
                self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def build(self, chunks: List[Chunk]) -> None:
        """Encode chunks and build FAISS IndexFlatIP (cosine similarity)."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            return

        texts = [f"{c.section_heading or ''}\n{c.text}".strip() for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Chunk, float, int]]:
        """
        Search dense FAISS index for relevant chunks.
        Returns list of (Chunk, score, rank_1based).
        """
        if not self.chunks or self.index is None:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results: List[Tuple[Chunk, float, int]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score), rank))

        return results
