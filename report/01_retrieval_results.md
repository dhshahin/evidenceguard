# Retrieval Study: Results and Analysis

*EvidenceGuard — Phase 1 (retrieval). Draft results section.*

## Summary

We compare four retrieval configurations for scientific evidence
synthesis over a corpus of five open-access breast-cancer machine-learning
papers (73 pages, 265 chunks). Retrieval is evaluated against a
hand-labelled benchmark of 11 research questions (9 answerable, 2
deliberately unanswerable for later abstention testing). We find that
embedding-based retrieval clearly outperforms keyword retrieval, that a
cross-encoder reranker further improves ranking quality (notably MRR),
and — through per-question diagnostics — we separate *ranking* failures
(fixable by reranking) from *recall* failures (not fixable by reranking).

These results are from a small pilot benchmark (9 answerable questions,
one corpus, one domain) and are therefore directional rather than
definitive. They are reported here to establish the experimental pipeline
and the analysis method; larger-scale confirmation is future work.

## Experimental setup

**Corpus.** Five open-access papers on machine learning for breast cancer
(recurrence, mortality, gene-expression, internal-mammary-node metastasis,
and a systematic review). Extracted with PyMuPDF, split into overlapping
word-window chunks (~220 words, 40-word overlap), preserving paper id,
title, and page number per chunk.

**Benchmark.** 11 questions labelled by hand. Each question specifies the
set of chunk ids that genuinely contain the answer. Relevant chunks were
found using retrieval-assisted shortlisting followed by a raw-text
"anti-bias" pass, so that relevant chunks the retriever *misses* are still
labelled (avoiding a benchmark that only credits what retrieval already
finds). Question types span single-fact, list/aggregation, cross-paper,
single-paper, and abstention. Two questions are deliberately unanswerable
(empty relevant set) and are excluded from retrieval metrics; they are
reserved for abstention evaluation in Phase 3.

**Methods compared.**
- **BM25** — keyword retrieval (rank-bm25).
- **Dense** — all-MiniLM-L6-v2 embeddings, FAISS cosine.
- **Hybrid** — Reciprocal Rank Fusion of BM25 + dense (rrf_k=60).
- **Rerank** — cross-encoder (ms-marco-MiniLM-L-6-v2) re-scoring a
  50-candidate hybrid pool.

**Metrics.** Recall@k (k = 5, 10, 20) and Mean Reciprocal Rank, averaged
over the 9 answerable questions.

## Aggregate results

| Method | Recall@5 | Recall@10 | Recall@20 | MRR |
|--------|:--------:|:---------:|:---------:|:---:|
| BM25   | 0.056 | 0.194 | 0.361 | 0.159 |
| Dense  | 0.250 | 0.361 | 0.546 | 0.224 |
| Hybrid | 0.259 | 0.343 | 0.556 | 0.227 |
| Rerank | 0.306 | 0.435 | 0.556 | 0.329 |

**Reading the table.**

1. **Embedding retrieval beats keyword retrieval decisively.** Dense and
   hybrid roughly quadruple BM25's Recall@5. BM25 does recover relevant
   chunks as k grows (0.056 → 0.361 at k=20) but ranks them poorly, which
   its low MRR (0.159) reflects.

2. **Hybrid gives no measurable gain over dense.** Recall and MRR are
   within one chunk on one question of each other. On this corpus,
   Hypothesis 1 ("hybrid > dense") is **not supported**. This is reported
   as a negative result rather than smoothed over.

3. **Reranking improves ranking quality.** Rerank gives the best Recall@5
   (0.306) and Recall@10 (0.435), and a large MRR gain (0.329 vs 0.227,
   +45% relative). Recall@20 is unchanged (0.556) because the reranker
   only reorders hybrid's candidate pool — it cannot add chunks the pool
   does not contain. The MRR jump indicates it moves the first relevant
   chunk substantially higher.

## Per-question diagnostics

Averages hide the mechanism. The table below (Recall@10, hybrid vs rerank)
shows where reranking helps and where it cannot.

| Question | #relevant | Hybrid | Rerank | Cause |
|----------|:---------:|:------:|:------:|-------|
| q001 | 4 | 0.50 | 0.75 | ranking — improved |
| q003 | 6 | 0.17 | 0.17 | partly unreachable |
| q006 | 3 | 0.00 | 0.00 | domain-semantic gap |
| q007 | 2 | 0.00 | 0.00 | **unreachable** in pool |
| q009 | 3 | 0.00 | 0.33 | ranking — improved |
| q010 | 2 | 0.50 | 0.50 | already correct |

We distinguish two failure types by checking, for each missed relevant
chunk, its rank in a deep (top-60) hybrid pool:

- **Ranking failures** — the chunk is in the pool but ranked low.
  Reranking can fix these (q009: 0.00 → 0.33; q001: 0.50 → 0.75).
- **Recall failures** — the chunk is absent from the pool entirely.
  Reranking is powerless (q007: both AUC chunks absent from top-60).

q007 is a clean control: because its answer chunks are unreachable,
reranking leaves it at 0.00, confirming that the reranker's benefit is
about ordering, not retrieval coverage.

## Error analysis: two illustrative failures

**q007 — recall failure (vocabulary mismatch, extreme).** The question
asks the model's AUC. The answer chunk reads *"Table 5 and Supplementary
Fig. 4, showing mean AUCs 0.950..."* — dominated by table/figure
references. Neither dense nor hybrid places it in the top 60, so it is
never a rerank candidate. The fix must be upstream: query expansion,
domain-adapted embeddings, or chunking that keeps result statements intact.

**q006 — reranker semantic gap (domain inference).** The question asks
which genes were most important. The cross-encoder assigned its *highest*
score (5.14) to a chunk echoing the query's words — *"challenging to
pinpoint... the most influential [genes]"* — that names no genes, while
penalising the chunk that actually lists them via LOCI ranking (score
−0.24, rank 13) and the top-four-genes chunk (−4.13, rank 37). A
general-domain reranker rewards surface query–passage overlap and cannot
bridge *"most important predictors" → specific gene symbols*. This
motivates domain-adapted rerankers or query expansion as future work.

## Conclusions

- Embedding-based retrieval substantially outperforms BM25 on this corpus.
- Hybrid fusion did not beat dense retrieval alone here (negative result).
- Cross-encoder reranking improves ranking (MRR +45%, higher Recall@5/10)
  but cannot improve coverage beyond the retrieved pool.
- Per-question diagnostics separate ranking failures (rerankable) from
  recall failures (not rerankable), and error analysis identifies
  vocabulary mismatch and domain-semantic gaps as the dominant causes.

## Limitations

- **Small benchmark** — 9 answerable questions; numbers are directional.
  The reranker's MRR advantage in particular needs confirmation at n≥30.
- **Single corpus, single domain** — results may not transfer.
- **Single annotator** — no inter-annotator agreement; relevance
  judgments are one author's. A re-labelling consistency check is planned.
- **Hard-positive labelling** — some relevant chunks were found by manual
  search and are hard for any method to retrieve, making the benchmark
  demanding at low k. Multi-k reporting mitigates but does not remove this.

## Next steps

1. Grow the benchmark toward 30–50 questions and re-confirm the ranking.
2. Test query expansion / domain-adapted embeddings on recall failures.
3. Proceed to answer generation with forced citation and abstention
   (Phase 3), using q005 and q011 as the abstention tests.
