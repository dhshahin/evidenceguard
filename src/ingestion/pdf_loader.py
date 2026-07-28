"""Extract text + metadata from PDF papers.

Design choice: we extract PAGE BY PAGE and keep the page number, because
the whole project depends on being able to cite an exact source location.
Losing page numbers here would break citation accuracy downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Page:
    paper_id: str      # stable id, e.g. the filename stem
    title: str         # best-guess title (see _guess_title)
    page_number: int   # 1-indexed
    text: str


def _clean(text: str) -> str:
    """Light cleanup: collapse whitespace, drop hyphenation at line breaks."""
    text = re.sub(r"-\n(\w)", r"\1", text)      # join hyphen-split words
    text = re.sub(r"\s*\n\s*", " ", text)        # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)          # collapse runs of spaces
    return text.strip()


def _guess_title(doc: fitz.Document, fallback: str) -> str:
    """Try PDF metadata title; fall back to the filename stem.

    Deliberately simple. A fancier title extractor (largest font on page 1)
    is a nice-to-have, not now.
    """
    meta_title = (doc.metadata or {}).get("title") or ""
    meta_title = meta_title.strip()
    if len(meta_title) >= 5:
        return meta_title
    return fallback


def load_pdf(path: str | Path) -> list[Page]:
    """Return one Page per PDF page."""
    path = Path(path)
    paper_id = path.stem
    doc = fitz.open(path)
    title = _guess_title(doc, fallback=paper_id)

    pages: list[Page] = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = _clean(raw)
        if cleaned:  # skip blank pages
            pages.append(Page(paper_id=paper_id, title=title,
                              page_number=i, text=cleaned))
    doc.close()
    return pages


def load_corpus(raw_dir: str | Path) -> list[Page]:
    """Load every PDF in a directory."""
    raw_dir = Path(raw_dir)
    pdfs = sorted(raw_dir.glob("*.pdf"))
    all_pages: list[Page] = []
    for pdf in pdfs:
        all_pages.extend(load_pdf(pdf))
    return all_pages


def page_to_dict(p: Page) -> dict:
    return asdict(p)
