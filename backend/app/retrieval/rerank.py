import os
from typing import Dict, List, Optional, Tuple
from backend.app.schemas import Chunk, ChunkKind

MAX_METHOD_CONTEXT_CHARS = 300
EVAL_HEADING_KEYWORDS = ("evaluation", "experiment", "implementation", "setup")


def get_methodological_context(
    chunk: Chunk,
    doc_chunks: Optional[List[Chunk]] = None,
    max_context_chars: int = MAX_METHOD_CONTEXT_CHARS,
) -> str:
    """
    Select bounded methodological context from preceding sections within the same document.
    Appended to implementation/evaluation chunks at inference time to bridge cross-passage split evidence.
    Returns empty string if chunk is not under an empirical section or no preceding method context exists.
    """
    if not chunk.section_heading:
        return ""

    heading_lower = chunk.section_heading.strip().lower()
    if not any(kw in heading_lower for kw in EVAL_HEADING_KEYWORDS):
        return ""

    if doc_chunks is None:
        try:
            from backend.app.core.storage import storage

            doc_chunks = storage.get_chunks(chunk.doc_id)
        except Exception:
            return ""

    if not doc_chunks:
        return ""

    current_idx = -1
    for idx, c in enumerate(doc_chunks):
        if c.chunk_id == chunk.chunk_id:
            current_idx = idx
            break

    if current_idx <= 0:
        return ""

    top_section_num = ""
    first_token = chunk.section_heading.strip().split()[0]
    if first_token.endswith(".") and first_token[:-1].isdigit():
        top_section_num = first_token[:-1]

    for past_idx in range(current_idx - 1, -1, -1):
        past_chunk = doc_chunks[past_idx]
        if past_chunk.doc_id != chunk.doc_id:
            continue
        if past_chunk.kind in (ChunkKind.REFERENCES, ChunkKind.TABLE):
            continue

        past_heading = (past_chunk.section_heading or "").strip()
        if not past_heading:
            continue

        past_heading_lower = past_heading.lower()
        if any(kw in past_heading_lower for kw in EVAL_HEADING_KEYWORDS):
            continue
        if top_section_num and past_heading.startswith(f"{top_section_num}."):
            continue

        clean_text = past_chunk.text.strip()
        if len(clean_text) > max_context_chars:
            clean_snippet = clean_text[-max_context_chars:].strip()
            first_space = clean_snippet.find(" ")
            if 0 <= first_space < 30:
                clean_snippet = clean_snippet[first_space + 1 :]
        else:
            clean_snippet = clean_text

        return f"\n[Method Context: {past_heading} - {clean_snippet}]"

    return ""


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
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: List[Chunk], top_k: int = 10
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Score (query, chunk_text) pairs and return top_k reranked candidates.
        Appends bounded methodological context from preceding sections when candidate
        appears under an empirical/evaluation heading, preserving the original candidate identity.
        Returns list of (Chunk, rerank_score, rank) with 1-indexed rank.
        """
        if not candidates:
            return []

        from backend.app.ingestion.chunker import chunk_index_text

        # Cache document chunks across candidate batch to minimize storage lookup overhead
        doc_chunks_cache: Dict[str, List[Chunk]] = {}
        for c in candidates:
            if c.doc_id not in doc_chunks_cache:
                try:
                    from backend.app.core.storage import storage

                    doc_chunks_cache[c.doc_id] = storage.get_chunks(c.doc_id)
                except Exception:
                    doc_chunks_cache[c.doc_id] = []

        pairs = []
        for c in candidates:
            base_passage = chunk_index_text(c)
            context = get_methodological_context(
                c, doc_chunks=doc_chunks_cache.get(c.doc_id, [])
            )
            pairs.append([query, base_passage + context])

        scores = self.model.predict(pairs, show_progress_bar=False)

        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[Tuple[Chunk, float, int]] = []
        for rank, (chunk, score) in enumerate(scored_candidates[:top_k], start=1):
            results.append((chunk, float(score), rank))

        return results
