"""BM25 keyword retrieval — the keyword baseline.

BM25 ranks chunks by term overlap with the query, weighted by term
rarity and chunk length. No embeddings, no GPU, no model download.
It is a genuinely strong baseline on small technical corpora, which is
exactly why comparing against it matters.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.retrieval.base import Retriever, RetrievalResult


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Simple and reproducible."""
    return _TOKEN.findall(text.lower())


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        tokenized = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before search().")
        scores = self._bm25.get_scores(tokenize(query))
        return self._to_results(self._chunks, scores, k)
