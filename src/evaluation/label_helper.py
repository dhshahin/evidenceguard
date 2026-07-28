"""Benchmark labeling helper.

Given a question, retrieve the top-N candidate chunks and print them with
their chunk_ids, so you can eyeball a shortlist (not all 265 chunks) and
copy the truly-relevant chunk_ids into data/evaluation/benchmark.jsonl.

Usage inside a notebook after building bm25/dense/hybrid:

    from src.evaluation.label_helper import show_candidates
    show_candidates("Which features predicted recurrence?", hybrid, n=15)

Then hand-pick the relevant ids and append a benchmark row with:

    from src.evaluation.label_helper import make_row
    print(make_row(qid="q001",
                   question="...",
                   answer="...",
                   relevant_chunk_ids=["cmar-17-917::p9::c0"],
                   supporting_papers=["cmar-17-917"]))
"""
from __future__ import annotations

import json


def show_candidates(question: str, retriever, n: int = 15, width: int = 160) -> None:
    """Print top-n candidate chunks with ids for manual relevance labeling."""
    print("Q:", question)
    print("=" * width)
    for res in retriever.search(question, k=n):
        snippet = res.text[:width].replace("\n", " ").strip()
        print(f"[{res.rank:>2}] {res.chunk_id}")
        print(f"     ({res.paper_id} p{res.page_number})  {snippet}")
        print()


def make_row(qid: str, question: str, answer: str,
             relevant_chunk_ids: list[str],
             supporting_papers: list[str],
             notes: str = "") -> str:
    """Return a JSONL line ready to append to benchmark.jsonl."""
    row = {
        "qid": qid,
        "question": question,
        "answer": answer,
        "relevant_chunk_ids": relevant_chunk_ids,
        "supporting_papers": supporting_papers,
        "notes": notes,
    }
    return json.dumps(row, ensure_ascii=False)


def append_row(path: str, row_json: str) -> None:
    """Append one JSONL row to the benchmark file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(row_json + "\n")
