from typing import Dict, List, Optional, Set, Tuple
from backend.app.schemas import Chunk, ChunkKind, ScoredChunk
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.rerank import CrossEncoderReranker


def select_allowed_candidates(
    raw_results: List[Tuple[Chunk, float, int]],
    max_candidates: int,
    include_references: bool = False,
    doc_ids: Optional[Set[str]] = None,
    raw_depth: Optional[int] = None,
) -> List[Tuple[Chunk, float, int]]:
    """
    Select allowed candidates from raw retrieval results up to raw_depth (default: max_candidates * 2),
    filtering out references (if include_references is False) and restricted doc_ids,
    preserving original retrieval ranks, and capping at max_candidates.
    When multiple doc_ids are specified, provides balanced candidate allocation across requested documents.
    """
    if raw_depth is None:
        raw_depth = (
            max(max_candidates * 2, max_candidates * len(doc_ids) * 2)
            if doc_ids and len(doc_ids) >= 2
            else max_candidates * 2
        )
    allowed_pool: List[Tuple[Chunk, float, int]] = []
    for chunk, score, rank in raw_results[:raw_depth]:
        if not include_references and chunk.kind == ChunkKind.REFERENCES:
            continue
        if doc_ids is not None and chunk.doc_id not in doc_ids:
            continue
        allowed_pool.append((chunk, score, rank))

    # Single-document or unrestricted: standard sequential selection preserving original behavior
    if not doc_ids or len(doc_ids) <= 1:
        return allowed_pool[:max_candidates]

    # Multi-document: balanced candidate allocation per requested document
    by_doc: Dict[str, List[Tuple[Chunk, float, int]]] = {}
    for item in allowed_pool:
        by_doc.setdefault(item[0].doc_id, []).append(item)

    per_doc_quota = max(1, max_candidates // len(doc_ids))
    selected: List[Tuple[Chunk, float, int]] = []
    selected_cids: Set[str] = set()

    # Pass 1: select top candidates up to per_doc_quota from each requested document
    for d in doc_ids:
        doc_items = by_doc.get(d, [])
        for item in doc_items[:per_doc_quota]:
            selected.append(item)
            selected_cids.add(item[0].chunk_id)

    # Pass 2: fill remaining quota with highest-ranked remaining items across all docs
    for item in allowed_pool:
        if len(selected) >= max_candidates:
            break
        if item[0].chunk_id not in selected_cids:
            selected.append(item)
            selected_cids.add(item[0].chunk_id)

    # Sort selected candidates by their original rank to preserve monotonic ordering
    selected.sort(key=lambda x: x[2])
    return selected


class HybridRetriever:
    """
    Hybrid retriever combining BM25 lexical search and FAISS dense vector search
    via Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking.
    """

    def __init__(
        self,
        dense_model_name: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
    ):
        self.bm25 = BM25Index()
        self.dense = DenseIndex(model_name=dense_model_name)
        self.reranker = CrossEncoderReranker(model_name=reranker_model_name)
        self.chunks_by_id: Dict[str, Chunk] = {}
        self.last_total_candidates: int = 0

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Index or update chunk corpus for both BM25 and dense retrieval."""
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self.bm25.build(chunks)
        self.dense.build(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
        include_references: bool = False,
        use_reranker: bool = True,
        candidate_pool_size: Optional[int] = None,
    ) -> List[ScoredChunk]:
        """
        Execute hybrid search:
        1. BM25 search -> top N
        2. Dense search -> top N
        3. Reciprocal Rank Fusion (RRF) -> top candidate_pool_size (default from RERANK_POOL or 30)
        4. Cross-Encoder reranking -> top_k
        Filters out ChunkKind.REFERENCES by default.
        """
        if not self.chunks_by_id:
            return []

        if candidate_pool_size is None:
            import os
            candidate_pool_size = int(os.environ.get("RERANK_POOL", "30"))

        doc_filter_set: Optional[Set[str]] = set(doc_ids) if doc_ids is not None else None
        if doc_filter_set is not None and len(doc_filter_set) == 0:
            self.last_total_candidates = 0
            return []

        def is_allowed(chunk: Chunk) -> bool:
            if not include_references and chunk.kind == ChunkKind.REFERENCES:
                return False
            if doc_filter_set is not None and chunk.doc_id not in doc_filter_set:
                return False
            return True

        # Multi-document candidate scaling: When multiple documents are requested, ensure
        # each requested document receives a fair allocation of candidates (up to candidate_pool_size
        # per document) before RRF fusion, preventing specific technical evidence in one document
        # from being crowded out by high-frequency overview terms in another.
        effective_pool_size = (
            candidate_pool_size * len(doc_filter_set)
            if doc_filter_set and len(doc_filter_set) >= 2
            else candidate_pool_size
        )

        # Stage 1: BM25 retrieval
        raw_search_depth = (
            max(effective_pool_size * 2, effective_pool_size * len(doc_filter_set) * 2)
            if doc_filter_set and len(doc_filter_set) >= 2
            else effective_pool_size * 2
        )
        bm25_raw = self.bm25.search(query, top_k=raw_search_depth)
        bm25_results = select_allowed_candidates(
            bm25_raw,
            max_candidates=effective_pool_size,
            include_references=include_references,
            doc_ids=doc_filter_set,
            raw_depth=raw_search_depth,
        )
        bm25_dict = {c.chunk_id: (score, rank) for c, score, rank in bm25_results}

        # Stage 2: Dense retrieval
        dense_raw = self.dense.search(query, top_k=raw_search_depth)
        dense_results = select_allowed_candidates(
            dense_raw,
            max_candidates=effective_pool_size,
            include_references=include_references,
            doc_ids=doc_filter_set,
            raw_depth=raw_search_depth,
        )
        dense_dict = {c.chunk_id: (score, rank) for c, score, rank in dense_results}

        # Format rankings for RRF
        bm25_rank_tuples = [(c.chunk_id, rank) for c, _, rank in bm25_results]
        dense_rank_tuples = [(c.chunk_id, rank) for c, _, rank in dense_results]

        # Stage 3: Score Fusion via RRF
        fused = reciprocal_rank_fusion(
            rankings=[bm25_rank_tuples, dense_rank_tuples],
            k=60,
        )

        fused_filtered = [(cid, score) for cid, score in fused if cid in self.chunks_by_id and is_allowed(self.chunks_by_id[cid])]

        if doc_filter_set and len(doc_filter_set) >= 2:
            by_doc_fused: Dict[str, List[Tuple[str, float]]] = {}
            for cid, score in fused_filtered:
                d = self.chunks_by_id[cid].doc_id
                by_doc_fused.setdefault(d, []).append((cid, score))

            per_doc_fused = max(1, effective_pool_size // len(doc_filter_set))
            fused_pool_cids: List[str] = []
            seen_fused_cids: Set[str] = set()

            for d in doc_filter_set:
                for cid, score in by_doc_fused.get(d, [])[:per_doc_fused]:
                    fused_pool_cids.append(cid)
                    seen_fused_cids.add(cid)

            for cid, score in fused_filtered:
                if len(fused_pool_cids) >= effective_pool_size:
                    break
                if cid not in seen_fused_cids:
                    fused_pool_cids.append(cid)
                    seen_fused_cids.add(cid)

            fused_filtered_map = dict(fused_filtered)
            fused_pool = [(cid, fused_filtered_map[cid]) for cid in fused_pool_cids]
            fused_pool.sort(key=lambda x: x[1], reverse=True)
        else:
            fused_pool = fused_filtered[:effective_pool_size]

        fused_dict: Dict[str, Tuple[float, int]] = {}
        candidate_chunks: List[Chunk] = []
        for rank, (cid, score) in enumerate(fused_pool, start=1):
            fused_dict[cid] = (score, rank)
            candidate_chunks.append(self.chunks_by_id[cid])

        self.last_total_candidates = len(candidate_chunks)

        # Stage 4: Cross-Encoder Reranker
        if use_reranker and candidate_chunks:
            rerank_k = len(candidate_chunks) if (doc_filter_set and len(doc_filter_set) >= 2) else top_k
            reranked = self.reranker.rerank(query, candidate_chunks, top_k=rerank_k)
            rerank_dict = {c.chunk_id: (score, rank) for c, score, rank in reranked}

            if doc_filter_set and len(doc_filter_set) >= 2 and top_k >= len(doc_filter_set):
                top_score = reranked[0][1] if reranked else 0.0
                # A chunk is eligible for the balanced allowance only if it meets a relevance standard:
                # it must have a positive rerank score or be within 6.0 logit units of the top score.
                relevance_threshold = max(0.0, top_score - 6.0)

                by_doc_relevant: Dict[str, List[Chunk]] = {}
                for c, score, _ in reranked:
                    if score >= relevance_threshold:
                        by_doc_relevant.setdefault(c.doc_id, []).append(c)

                # Small per-document evidence allowance (up to 2 chunks per qualified document)
                doc_allowance = min(2, max(1, top_k // (len(doc_filter_set) * 2))) if top_k >= 4 else 1
                balanced_chunks: List[Chunk] = []
                seen_final_cids: Set[str] = set()

                # Pass 1: Allocate allowance only to requested documents that have genuinely relevant evidence
                for d in doc_filter_set:
                    for c in by_doc_relevant.get(d, [])[:doc_allowance]:
                        balanced_chunks.append(c)
                        seen_final_cids.add(c.chunk_id)

                # Pass 2: Fill all remaining slots strictly by globally highest reranker scores
                for c, _, _ in reranked:
                    if len(balanced_chunks) >= top_k:
                        break
                    if c.chunk_id not in seen_final_cids:
                        balanced_chunks.append(c)
                        seen_final_cids.add(c.chunk_id)

                balanced_chunks.sort(key=lambda c: rerank_dict.get(c.chunk_id, (0.0, 0))[0], reverse=True)
                final_ordered_chunks = balanced_chunks
            else:
                final_ordered_chunks = [c for c, _, _ in reranked][:top_k]
        else:
            final_ordered_chunks = candidate_chunks[:top_k]
            rerank_dict = {}

        # Compile ScoredChunk objects
        scored_results: List[ScoredChunk] = []
        for chunk in final_ordered_chunks:
            cid = chunk.chunk_id
            b_score, b_rank = bm25_dict.get(cid, (None, None))
            d_score, d_rank = dense_dict.get(cid, (None, None))
            f_score, f_rank = fused_dict.get(cid, (None, None))
            r_score, r_rank = rerank_dict.get(cid, (None, None))

            scored_results.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page=chunk.page,
                    section_heading=chunk.section_heading,
                    text=chunk.text,
                    kind=chunk.kind,
                    bm25_score=b_score,
                    bm25_rank=b_rank,
                    dense_score=d_score,
                    dense_rank=d_rank,
                    fused_score=f_score,
                    fused_rank=f_rank,
                    rerank_score=r_score,
                    rerank_rank=r_rank,
                )
            )

        return scored_results


_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


class _LazyRetrieverProxy:
    """Proxy object that defers HybridRetriever creation until first method call."""

    def __getattr__(self, name):
        return getattr(get_retriever(), name)


retriever = _LazyRetrieverProxy()


def ensure_retriever_indexed(force: bool = False) -> int:
    """
    Ensure the global retriever has loaded all chunks from persistent storage.
    If the retriever is already indexed and force is False, does not re-index.
    Returns the number of chunks indexed.
    """
    if not force and bool(retriever.chunks_by_id):
        return len(retriever.chunks_by_id)

    from backend.app.core.storage import storage

    all_chunks = storage.get_all_chunks()
    if all_chunks:
        retriever.index_chunks(all_chunks)
        return len(all_chunks)
    return 0
