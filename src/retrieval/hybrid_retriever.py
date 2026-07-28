"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Runs BM25 and dense retrieval, then fuses their ranked lists. RRF combines
by RANK POSITION, not raw score, so it sidesteps the fact that BM25 scores
(0..~20) and cosine scores (0..1) live on incompatible scales. Each chunk
gets sum over methods of 1 / (rrf_k + rank). rrf_k=60 is the standard
default from the original RRF paper; keeping it fixed means there is no
tuned hyperparameter to justify.

Reference: Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet
and Individual Rank Learning Methods" (SIGIR 2009).
"""
from __future__ import annotations

from src.retrieval.base import Retriever, RetrievalResult


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, bm25: Retriever, dense: Retriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self._chunks_by_id: dict[str, dict] = {}

    def index(self, chunks: list[dict]) -> None:
        # Assumes bm25 and dense are already indexed on the same chunks.
        self._chunks_by_id = {c["chunk_id"]: c for c in chunks}

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        # Pull deeper than k from each method so fusion has material to work with.
        pool = max(k * 4, 20)
        bm = self.bm25.search(query, k=pool)
        de = self.dense.search(query, k=pool)

        fused: dict[str, float] = {}
        for results in (bm, de):
            for res in results:
                fused[res.chunk_id] = fused.get(res.chunk_id, 0.0) + \
                    1.0 / (self.rrf_k + res.rank)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            c = self._chunks_by_id[chunk_id]
            out.append(RetrievalResult(
                chunk_id=chunk_id, paper_id=c["paper_id"], title=c["title"],
                page_number=c["page_number"], text=c["text"],
                score=float(score), rank=rank))
        return out
