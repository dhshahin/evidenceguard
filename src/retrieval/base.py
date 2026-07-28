"""Shared retrieval types and the common Retriever interface.

Every retrieval method (BM25, dense, hybrid) subclasses Retriever and
implements .search(). Because they all return the same RetrievalResult
objects, the evaluation harness and the generator never need to know
which method produced them. This is the key design decision that makes
the A/B/C/D experiment clean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalResult:
    chunk_id: str
    paper_id: str
    title: str
    page_number: int
    text: str
    score: float      # method-specific score; higher = more relevant
    rank: int         # 1-indexed position in this result list


def load_chunks(path: str | Path) -> list[dict]:
    """Load the chunks.jsonl produced by scripts_build_chunks.py."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Retriever:
    """Base class. Subclasses implement index() and search()."""

    name: str = "base"

    def index(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError

    def _to_results(self, chunks, scores, k) -> list[RetrievalResult]:
        """Helper: sort chunks by score desc, take top-k, wrap in results."""
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for rank, i in enumerate(order[:k], start=1):
            c = chunks[i]
            results.append(RetrievalResult(
                chunk_id=c["chunk_id"], paper_id=c["paper_id"],
                title=c["title"], page_number=c["page_number"],
                text=c["text"], score=float(scores[i]), rank=rank,
            ))
        return results
