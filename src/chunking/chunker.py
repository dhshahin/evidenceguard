"""Split pages into retrieval chunks while preserving metadata.

Each chunk carries everything needed to cite it later: paper_id, title,
page_number, and a unique chunk_id. This metadata is the backbone of the
whole "trustworthy citation" claim.

Strategy: fixed-size overlapping word windows. It is simple, reproducible,
and language-agnostic. Semantic/section-aware chunking is a later upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from src.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
from src.ingestion.pdf_loader import Page


@dataclass
class Chunk:
    chunk_id: str      # e.g. "paperA::p3::c2"
    paper_id: str
    title: str
    page_number: int
    text: str


def _window(words: list[str], size: int, overlap: int):
    """Yield overlapping slices of the word list."""
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            yield chunk
        if start + size >= len(words):
            break


def chunk_page(page: Page,
               size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    words = page.text.split()
    chunks: list[Chunk] = []
    for ci, w in enumerate(_window(words, size, overlap)):
        cid = f"{page.paper_id}::p{page.page_number}::c{ci}"
        chunks.append(Chunk(
            chunk_id=cid,
            paper_id=page.paper_id,
            title=page.title,
            page_number=page.page_number,
            text=" ".join(w),
        ))
    return chunks


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        out.extend(chunk_page(p))
    return out


def chunk_to_dict(c: Chunk) -> dict:
    return asdict(c)
