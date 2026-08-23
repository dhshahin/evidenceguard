"""Reproduce the generation-phase evaluation.

Prerequisites:
    python scripts_build_chunks.py     # produces data/processed/chunks.jsonl
    export ANTHROPIC_API_KEY=sk-...    # generation calls the Claude API

Then:
    python run_generation_eval.py

Pipeline per question:
    retrieve (rerank) -> abstention gate -> generate (cite or abstain)
    -> confidence -> score citation validity / grounding / abstention.

Uses the rerank retriever (best from the retrieval study) as the evidence
source. Prints citation, grounding, abstention, and confidence summaries.
"""
from __future__ import annotations

import torch

from src.config import PROCESSED, EVAL, TOP_K
from src.retrieval.base import load_chunks
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rerank_retriever import RerankRetriever
from src.evaluation.metrics import load_benchmark
from src.generation.generator import generate_answer
from src.generation.abstention import (
    top_score, should_abstain_on_retrieval, suggest_threshold,
)
from src.confidence.confidence import score_confidence
from src.evaluation.generation_metrics import (
    citation_validity, citation_grounding, AbstentionCounts, tally_abstention,
)
from anthropic import Anthropic


def build_retriever(chunks, device):
    bm25 = BM25Retriever(); bm25.index(chunks)
    dense = DenseRetriever(device=device); dense.index(chunks)
    hybrid = HybridRetriever(bm25, dense); hybrid.index(chunks)
    return RerankRetriever(hybrid, pool=50, device=device)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    chunks = load_chunks(PROCESSED / "chunks.jsonl")
    benchmark = load_benchmark(EVAL / "benchmark.jsonl")
    retriever = build_retriever(chunks, device)
    client = Anthropic()

    # --- Pass 1: gather top-scores to set the abstention threshold + confidence range
    ans_scores, abs_scores, all_scores = [], [], []
    retrieved_cache = {}
    for q in benchmark:
        res = retriever.search(q["question"], k=TOP_K)
        retrieved_cache[q["qid"]] = res
        ts = top_score(res)
        all_scores.append(ts)
        (ans_scores if q["relevant_chunk_ids"] else abs_scores).append(ts)

    threshold = suggest_threshold(ans_scores, abs_scores)
    score_lo, score_hi = min(all_scores), max(all_scores)
    print(f"Abstention threshold (from calibration): {threshold:.3f}")
    print(f"Top-score range for confidence: [{score_lo:.3f}, {score_hi:.3f}]\n")

    # --- Pass 2: generate + evaluate
    counts = AbstentionCounts()
    validities, groundings, confidences = [], [], []

    print(f"{'qid':<6}{'abst':>6}{'cites':>7}{'valid':>7}{'ground':>7}{'conf':>7}")
    print("-" * 40)
    for q in benchmark:
        res = retrieved_cache[q["qid"]]
        is_answerable = bool(q["relevant_chunk_ids"])

        # retrieval abstention gate
        if should_abstain_on_retrieval(res, threshold):
            from src.generation.generator import GeneratedAnswer, ABSTAIN_MARKER
            ans = GeneratedAnswer(
                question=q["question"], answer=ABSTAIN_MARKER, abstained=True,
                passages_used=[r.chunk_id for r in res])
        else:
            ans = generate_answer(q["question"], res, client=client)

        tally_abstention(ans, is_answerable, counts)
        conf = score_confidence(ans, res, score_lo, score_hi)
        confidences.append(conf.score)

        v = citation_validity(ans)
        g = citation_grounding(ans, q["relevant_chunk_ids"])
        if v is not None:
            validities.append(v)
        if g is not None:
            groundings.append(g)

        print(f"{q['qid']:<6}{str(ans.abstained):>6}{len(ans.cited_chunk_ids):>7}"
              f"{(f'{v:.2f}' if v is not None else '  -'):>7}"
              f"{(f'{g:.2f}' if g is not None else '  -'):>7}"
              f"{conf.score:>7.2f}")

    print("\n=== Summary ===")
    if validities:
        print(f"Mean citation validity:  {sum(validities)/len(validities):.2f}  "
              f"(fraction of cited ids that were real)")
    if groundings:
        print(f"Mean citation grounding: {sum(groundings)/len(groundings):.2f}  "
              f"(answered Qs citing a benchmark-relevant chunk)")
    print(f"Mean confidence (answered): "
          f"{sum(confidences)/len(confidences):.2f}")
    print()
    print(counts.as_table())


if __name__ == "__main__":
    main()
