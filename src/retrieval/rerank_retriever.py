"""Cross-encoder reranking (config C).

Wraps a base retriever: pulls a deep candidate pool, then re-scores every
candidate with a cross-encoder that reads (query, chunk) jointly. Unlike
the bi-encoder used for dense retrieval, the cross-encoder sees both texts
together, so it ranks relevance more precisely. It can only reorder the
pool the base retriever returns — it cannot add chunks the pool lacks.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL
from src.retrieval.base import Retriever, RetrievalResult


class RerankRetriever(Retriever):
    name = "rerank"

    def __init__(self, base: Retriever, model_name: str = RERANKER_MODEL,
                 pool: int = 50, device: str | None = None):
        self.base = base
        self.ce = CrossEncoder(model_name, device=device)
        self.pool = pool

    def index(self, chunks: list[dict]) -> None:
        pass  # the base retriever is already indexed

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        cands = self.base.search(query, k=self.pool)
        if not cands:
            return []
        scores = self.ce.predict([(query, c.text) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        out = []
        for rank, i in enumerate(order[:k], start=1):
            c = cands[i]
            out.append(RetrievalResult(
                chunk_id=c.chunk_id, paper_id=c.paper_id, title=c.title,
                page_number=c.page_number, text=c.text,
                score=float(scores[i]), rank=rank))
        return out
