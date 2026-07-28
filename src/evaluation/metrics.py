"""Retrieval evaluation metrics and harness.

All metrics return None for questions with no relevant chunks (the
abstention questions), which are excluded from retrieval scoring and
evaluated separately in the generation phase.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_benchmark(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(retrieved: list[str], relevant: list[str], k: int):
    if not relevant:
        return None
    topk = retrieved[:k]
    return sum(1 for r in relevant if r in topk) / len(relevant)


def mrr(retrieved: list[str], relevant: list[str]):
    if not relevant:
        return None
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int):
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 2)
              for i, c in enumerate(retrieved[:k]) if c in relevant)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate_multi_k(retriever, benchmark: list[dict], ks=(5, 10, 20)) -> list[dict]:
    rows = []
    for q in benchmark:
        rel = q["relevant_chunk_ids"]
        if not rel:
            continue
        retrieved = [r.chunk_id for r in retriever.search(q["question"], k=max(ks))]
        row = {"qid": q["qid"], "n_rel": len(rel),
               "mrr": mrr(retrieved, rel)}
        for k in ks:
            row[f"R@{k}"] = recall_at_k(retrieved, rel, k)
        rows.append(row)
    return rows


def comparison_table(retrievers, benchmark, ks=(5, 10, 20)) -> str:
    """Return a formatted multi-k comparison table across methods."""
    header = f"{'method':<8}" + "".join(f"{'R@'+str(k):>8}" for k in ks) + f"{'MRR':>8}"
    lines = [header, "=" * len(header)]
    for retr in retrievers:
        rows = evaluate_multi_k(retr, benchmark, ks)
        n = len(rows)
        line = f"{retr.name:<8}"
        for k in ks:
            line += f"{sum(r[f'R@{k}'] for r in rows)/n:>8.3f}"
        line += f"{sum(r['mrr'] for r in rows)/n:>8.3f}"
        lines.append(line)
    return "\n".join(lines)
