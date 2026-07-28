"""Dense (embedding-based) retrieval.

Encodes every chunk into a normalized vector with a sentence-transformer,
stores them in a FAISS inner-product index (== cosine on normalized
vectors), and ranks by semantic similarity. Complements BM25: it finds
passages that are about the query even when they share few exact words.
"""
from __future__ import annotations

import faiss
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.retrieval.base import Retriever, RetrievalResult


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
        self._chunks: list[dict] = []
        self._index = None

    def index(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        texts = [c["text"] for c in chunks]
        emb = self.model.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("Call index() before search().")
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        sims, idxs = self._index.search(q, k)
        results = []
        for rank, (i, s) in enumerate(zip(idxs[0], sims[0]), start=1):
            c = self._chunks[int(i)]
            results.append(RetrievalResult(
                chunk_id=c["chunk_id"], paper_id=c["paper_id"],
                title=c["title"], page_number=c["page_number"],
                text=c["text"], score=float(s), rank=rank))
        return results
