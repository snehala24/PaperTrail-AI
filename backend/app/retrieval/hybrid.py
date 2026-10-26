from typing import Dict, List, Optional, Set, Tuple
from backend.app.schemas import Chunk, ChunkKind, ScoredChunk
from backend.app.retrieval.bm25 import BM25Index
from backend.app.retrieval.dense import DenseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.rerank import CrossEncoderReranker


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
        candidate_pool_size: int = 30,
    ) -> List[ScoredChunk]:
        """
        Execute hybrid search:
        1. BM25 search -> top N
        2. Dense search -> top N
        3. Reciprocal Rank Fusion (RRF) -> top candidate_pool_size
        4. Cross-Encoder reranking -> top_k
        Filters out ChunkKind.REFERENCES by default.
        """
        if not self.chunks_by_id:
            return []

        doc_filter_set: Optional[Set[str]] = set(doc_ids) if doc_ids else None

        def is_allowed(chunk: Chunk) -> bool:
            if not include_references and chunk.kind == ChunkKind.REFERENCES:
                return False
            if doc_filter_set and chunk.doc_id not in doc_filter_set:
                return False
            return True

        # Stage 1: BM25 retrieval
        bm25_raw = self.bm25.search(query, top_k=candidate_pool_size * 2)
        bm25_results = [(c, score, rank) for c, score, rank in bm25_raw if is_allowed(c)]
        bm25_dict = {c.chunk_id: (score, rank) for c, score, rank in bm25_results}

        # Stage 2: Dense retrieval
        dense_raw = self.dense.search(query, top_k=candidate_pool_size * 2)
        dense_results = [(c, score, rank) for c, score, rank in dense_raw if is_allowed(c)]
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
        fused_pool = fused_filtered[:candidate_pool_size]

        fused_dict: Dict[str, Tuple[float, int]] = {}
        candidate_chunks: List[Chunk] = []
        for rank, (cid, score) in enumerate(fused_pool, start=1):
            fused_dict[cid] = (score, rank)
            candidate_chunks.append(self.chunks_by_id[cid])

        # Stage 4: Cross-Encoder Reranker
        if use_reranker and candidate_chunks:
            reranked = self.reranker.rerank(query, candidate_chunks, top_k=top_k)
            rerank_dict = {c.chunk_id: (score, rank) for c, score, rank in reranked}
            final_ordered_chunks = [c for c, _, _ in reranked]
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
