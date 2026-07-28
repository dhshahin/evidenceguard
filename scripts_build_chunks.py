"""Run the ingestion + chunking pipeline.

Usage (from the project root):
    python scripts_build_chunks.py

Reads every PDF in data/raw/, writes chunks to data/processed/chunks.jsonl,
and prints a summary. Run this whenever you add papers.
"""
import json
from pathlib import Path

from src.config import RAW, PROCESSED
from src.ingestion.pdf_loader import load_corpus
from src.chunking.chunker import chunk_pages, chunk_to_dict


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pages = load_corpus(RAW)
    if not pages:
        print(f"No PDFs found in {RAW}. Add some .pdf files and re-run.")
        return

    chunks = chunk_pages(pages)
    out_path = PROCESSED / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(chunk_to_dict(c), ensure_ascii=False) + "\n")

    papers = {p.paper_id for p in pages}
    print(f"Papers:  {len(papers)}")
    print(f"Pages:   {len(pages)}")
    print(f"Chunks:  {len(chunks)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
