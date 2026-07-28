"""Reproduce the full retrieval evaluation.

Prerequisites:
    python scripts_build_chunks.py     # produces data/processed/chunks.jsonl

Then:
    python run_retrieval_eval.py

Builds BM25, dense, hybrid, and rerank retrievers, evaluates them against
the benchmark, and prints the multi-k comparison table plus per-question
diagnostics. Uses GPU automatically if available.
"""
from __future__ import annotations

import torch

from src.config import PROCESSED, EVAL, TOP_K
from src.retrieval.base import load_chunks
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rerank_retriever import RerankRetriever
from src.evaluation.metrics import (
    load_benchmark, comparison_table, evaluate_multi_k,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    chunks = load_chunks(PROCESSED / "chunks.jsonl")
    benchmark = load_benchmark(EVAL / "benchmark.jsonl")
    n_answerable = sum(1 for q in benchmark if q["relevant_chunk_ids"])
    print(f"Chunks: {len(chunks)} | Benchmark: {len(benchmark)} "
          f"({n_answerable} answerable, {len(benchmark)-n_answerable} abstention)\n")

    # Build retrievers
    bm25 = BM25Retriever(); bm25.index(chunks)
    dense = DenseRetriever(device=device); dense.index(chunks)
    hybrid = HybridRetriever(bm25, dense); hybrid.index(chunks)
    rerank = RerankRetriever(hybrid, pool=50, device=device)

    # Aggregate comparison
    print("=== Aggregate retrieval comparison ===")
    print(comparison_table((bm25, dense, hybrid, rerank), benchmark))

    # Per-question diagnostics (hybrid vs rerank at k=10)
    print("\n=== Per-question Recall@10: hybrid vs rerank ===")
    print(f"{'qid':<6}{'n_rel':>6}{'hybrid':>8}{'rerank':>8}")
    h_rows = {r["qid"]: r for r in evaluate_multi_k(hybrid, benchmark)}
    rk_rows = {r["qid"]: r for r in evaluate_multi_k(rerank, benchmark)}
    for qid in sorted(h_rows):
        print(f"{qid:<6}{h_rows[qid]['n_rel']:>6}"
              f"{h_rows[qid]['R@10']:>8.2f}{rk_rows[qid]['R@10']:>8.2f}")


if __name__ == "__main__":
    main()
