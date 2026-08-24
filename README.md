[README (7).md](https://github.com/user-attachments/files/31370230/README.7.md)
# EvidenceGuard

> **Development status — research prototype**
>
> EvidenceGuard is an actively developed research prototype for trustworthy
> scientific evidence synthesis.
>
> The current public release contains the reproducible retrieval study
> (**Phase 1**). Citation-grounded generation, abstention, claim-level
> verification, confidence estimation, and the extended evaluation pipeline
> are undergoing final validation before release.
>
> Results reported here should therefore be interpreted as experimental
> research results rather than performance claims for a production system.

## Overview

EvidenceGuard investigates trustworthy and explainable Retrieval-Augmented
Generation (RAG) for scientific evidence synthesis.

The project studies whether retrieval quality, evidence attribution,
claim-level verification, abstention, and uncertainty signals can improve the
reliability of LLM-generated scientific answers.

> **Research question:** Can retrieval quality, evidence attribution, and
> uncertainty estimation improve the reliability of LLM-generated answers
> in scientific evidence synthesis?

## Current Status

### Phase 1 — Retrieval

- [x] PDF ingestion with page-level metadata
- [x] Overlapping chunking with stable chunk IDs
- [x] BM25 retrieval
- [x] Dense semantic retrieval
- [x] Hybrid retrieval using Reciprocal Rank Fusion
- [x] Cross-encoder reranking
- [x] Hand-labelled benchmark with 11 questions
- [x] Multi-k retrieval evaluation
- [x] Per-question error analysis
- [x] Reproducible retrieval pipeline

### Phase 2 — Trustworthy Generation

- [ ] Citation-grounded answer generation
- [ ] Retrieval-based abstention
- [ ] Claim-level evidence verification
- [ ] Confidence scoring
- [ ] Extended benchmark validation
- [ ] End-to-end generation evaluation
- [ ] Streamlit demonstration interface

Phase 2 is currently undergoing validation and is intentionally not presented
as a completed public result.

## Phase 1 Results

Retrieval performance on the pilot benchmark:

| Method | Recall@5 | Recall@10 | Recall@20 | MRR |
|--------|:--------:|:---------:|:---------:|:---:|
| BM25 | 0.056 | 0.194 | 0.361 | 0.159 |
| Dense | 0.250 | 0.361 | 0.546 | 0.224 |
| Hybrid | 0.259 | 0.343 | 0.556 | 0.227 |
| Rerank | 0.306 | 0.435 | 0.556 | 0.329 |

Dense retrieval substantially outperformed keyword-only retrieval on this
benchmark.

Cross-encoder reranking produced the strongest ranking quality, increasing MRR
from 0.227 for hybrid retrieval to 0.329.

The retrieval study also identified useful negative results. In particular,
hybrid retrieval did not consistently outperform dense retrieval at every
cutoff, illustrating why retrieval strategies should be evaluated empirically
rather than assumed to improve performance.

A detailed analysis is available in:

[`report/01_retrieval_results.md`](report/01_retrieval_results.md)

> **Evaluation note:** This is a small pilot benchmark based on one scientific
> domain and a limited corpus. Nine questions are answerable from the corpus
> and two are intentionally unanswerable. Results should be interpreted as
> directional evidence for system development rather than as general benchmark
> claims.

## Retrieval Pipeline

The current public pipeline is:

```text
Scientific PDFs
      |
      v
Page-level extraction
      |
      v
Overlapping chunks
      |
      +---------------------+
      |                     |
      v                     v
    BM25              Dense retrieval
      |                     |
      +----------+----------+
                 |
                 v
       Reciprocal Rank Fusion
                 |
                 v
        Cross-encoder reranking
                 |
                 v
          Ranked evidence
```

The next phase extends this pipeline with citation-constrained generation,
abstention, claim-level verification, and confidence estimation.

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/dhshahin/evidenceguard.git
cd evidenceguard
pip install -r requirements.txt
```

The retrieval phase does not require an LLM API key.

An Anthropic API key will be required for the generation phase when that
component is released:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Do not commit API keys or credentials to the repository.

## Reproduce the Retrieval Study

Place the open-access source PDFs in:

```text
data/raw/
```

Then build the corpus:

```bash
python scripts_build_chunks.py
```

Run the retrieval evaluation:

```bash
python run_retrieval_eval.py
```

The evaluation script builds the retrievers and reports Recall@k and MRR for
the benchmark questions.

## Repository Structure

```text
evidenceguard/
|
├── src/
│   ├── ingestion/          PDF -> pages and metadata
│   ├── chunking/           pages -> overlapping chunks
│   ├── retrieval/          BM25 / dense / hybrid / reranking
│   └── evaluation/         benchmark and retrieval metrics
│
├── data/
│   ├── raw/                source PDFs (gitignored)
│   ├── processed/          generated chunks (gitignored)
│   └── evaluation/         benchmark questions
│
├── report/
│   └── 01_retrieval_results.md
│
├── scripts_build_chunks.py
├── run_retrieval_eval.py
├── requirements.txt
└── README.md
```

Additional generation, verification, confidence, and testing components are
being validated before inclusion in the public release.

## Method Notes

### Stable evidence identifiers

Each chunk receives a stable identifier that preserves its relationship to the
source paper and page. This enables retrieved passages to be traced back to
their original evidence.

### Shared retrieval interface

All retrievers use a common `Retriever` interface and return
`RetrievalResult` objects. This allows the same evaluation harness to compare
different retrieval strategies.

### Dense retrieval

Dense retrieval uses sentence-transformer embeddings to retrieve passages by
semantic similarity rather than exact keyword overlap.

### Hybrid retrieval

Hybrid retrieval combines BM25 and dense rankings using Reciprocal Rank Fusion
with:

```text
rrf_k = 60
```

### Reranking

A cross-encoder reranks a candidate pool of retrieved passages:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The candidate pool contains the top 50 passages from the hybrid retrieval
stage.

### Benchmark construction

Relevant chunks were identified using an evidence-labelling process that was
not restricted to passages successfully retrieved by the system.

This reduces evaluation bias: a relevant passage remains part of the gold
evidence even when a retriever fails to retrieve it.

## Research Principles

EvidenceGuard is being developed around four principles:

1. **Traceability**  
   Scientific answers should be connected to identifiable source evidence.

2. **Abstention**  
   The system should prefer insufficient-evidence responses over unsupported
   answers.

3. **Claim-level verification**  
   Evidence support should be evaluated at the level of individual factual
   claims rather than only at the answer level.

4. **Transparent evaluation**  
   Retrieval, citation quality, verification, abstention, and confidence
   should be evaluated separately rather than collapsed into a single score.

## Limitations

The current public evaluation has several important limitations:

- small scientific corpus;
- only 11 benchmark questions;
- one primary scientific domain;
- manually constructed relevance labels;
- retrieval results have not yet been validated on a large external benchmark;
- generation and claim-verification results are still undergoing validation.

These limitations are intentionally documented because EvidenceGuard is a
research prototype rather than a production scientific evidence system.

## Planned Work

The next public release is planned to include:

- citation-constrained scientific answer generation;
- retrieval-based abstention;
- claim-level Natural Language Inference verification;
- evidence-window selection;
- terminology normalization;
- confidence scoring;
- audited benchmark annotations;
- end-to-end generation evaluation;
- automated tests;
- interactive demonstration interface.

## Reproducibility

The project is designed so that retrieval experiments can be rerun from the
source PDFs using the scripts included in this repository.

Generated data and source PDFs are intentionally excluded from version control
where appropriate.

## Disclaimer

EvidenceGuard is an experimental research project.

It is not intended to provide medical advice, clinical recommendations, or
automated scientific conclusions without expert review.
