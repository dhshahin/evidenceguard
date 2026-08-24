# EvidenceGuard

> **Development status — research prototype**
>
> EvidenceGuard is an actively developed research prototype for trustworthy
> scientific evidence synthesis. The public repository currently contains the
> reproducible retrieval study (Phase 1). The citation-grounded generation,
> claim-verification, abstention, and evaluation pipeline is undergoing final
> validation and will be released after the benchmark and verification audit
> is completed.
>
> Results reported in this repository should therefore be interpreted as
> experimental rather than as performance claims for a production system.

EvidenceGuard investigates trustworthy and explainable RAG for scientific
evidence synthesis. It studies whether retrieval quality, citation attribution,
claim-level verification, abstention, and uncertainty signals can improve the
reliability of LLM-generated scientific answers.

> **Research question:** Can retrieval quality, evidence attribution, and
> uncertainty estimation improve the reliability of LLM-generated answers
> in scientific evidence synthesis?

## Status

## Status

- [x] PDF ingestion (page-level, metadata preserved)
- [x] Chunking (overlapping word windows, stable chunk ids)
- [x] Retrieval: BM25 / dense / hybrid (RRF)
- [x] Cross-encoder reranking
- [x] Hand-labelled benchmark (11 questions) + evaluation harness
- [x] Retrieval study with multi-k metrics + error analysis
- [ ] Answer generation with forced citation + abstention
- [ ] Claim-level verification
- [ ] Confidence scoring
- [ ] Streamlit interface

## Key result (Phase 1: retrieval)

| Method | Recall@5 | Recall@10 | Recall@20 | MRR |
|--------|:--------:|:---------:|:---------:|:---:|
| BM25   | 0.056 | 0.194 | 0.361 | 0.159 |
| Dense  | 0.250 | 0.361 | 0.546 | 0.224 |
| Hybrid | 0.259 | 0.343 | 0.556 | 0.227 |
| Rerank | 0.306 | 0.435 | 0.556 | 0.329 |

Embedding retrieval clearly beats keyword retrieval; reranking improves
ranking quality (MRR +45%). Full analysis, including the hybrid-vs-dense
negative result and per-question error analysis, is in
[`report/01_retrieval_results.md`](report/01_retrieval_results.md).

*Pilot benchmark: 9 answerable questions, one corpus, one domain. Results
are directional; larger-scale confirmation is future work.*

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...      # for the generation phase (later)
```

## Reproduce

```bash
# 1. put open-access PDFs in data/raw/
python scripts_build_chunks.py       # PDFs -> data/processed/chunks.jsonl
python run_retrieval_eval.py         # builds retrievers, prints the table
```

## Layout

```
src/
  ingestion/    PDF -> pages (+metadata)
  chunking/     pages -> chunks
  retrieval/    bm25 / dense / hybrid / rerank (one shared interface)
  evaluation/   metrics + labelling helper
data/
  raw/          input PDFs (gitignored)
  processed/    chunks.jsonl (gitignored)
  evaluation/   benchmark.jsonl (the hand-labelled questions)
report/         written analysis
scripts_build_chunks.py    ingestion pipeline
run_retrieval_eval.py      full retrieval evaluation
```

## Method notes

- All retrievers share one `Retriever` interface returning `RetrievalResult`
  objects, so the evaluation harness is method-agnostic.
- Hybrid uses Reciprocal Rank Fusion (rrf_k=60) over BM25 + dense.
- Rerank uses a cross-encoder (ms-marco-MiniLM) over a 50-candidate pool.
- The benchmark labels relevant chunks found via an "anti-bias" raw-text
  pass, so chunks the retriever misses are still labelled — making the
  benchmark demanding at low k and honest about recall.
