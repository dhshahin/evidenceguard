[EvidenceGuard_README.md](https://github.com/user-attachments/files/31375633/EvidenceGuard_README.md)
# EvidenceGuard

**Trustworthy and Explainable RAG for Scientific Evidence Synthesis**

> **Status:** Research prototype — Phase 2 evaluated

EvidenceGuard is an experimental Retrieval-Augmented Generation (RAG) system for scientific evidence synthesis. It explores how evidence retrieval, citation attribution, abstention, claim-level verification, and transparent confidence signals can improve the reliability of LLM-generated scientific answers.

The project evaluates these components separately rather than collapsing reliability into a single score.

---

## Research Question

**Can retrieval quality, evidence attribution, claim-level verification, abstention, and uncertainty signals improve the reliability of LLM-generated answers in scientific evidence synthesis?**

---

## Project Status

### Phase 1 — Retrieval

- [x] PDF ingestion with page-level metadata
- [x] Stable overlapping chunking
- [x] BM25 retrieval
- [x] Dense semantic retrieval
- [x] Hybrid retrieval using Reciprocal Rank Fusion
- [x] Cross-encoder reranking
- [x] Retrieval benchmark and error analysis

### Phase 2 — Trustworthy Generation

- [x] Citation-grounded answer generation
- [x] Retrieval-based abstention
- [x] Claim-level evidence verification
- [x] Evidence-window construction
- [x] Terminology normalization
- [x] Post-verification claim filtering
- [x] Confidence scoring
- [x] Manually audited benchmark
- [x] End-to-end generation evaluation
- [x] Automated regression tests

### Planned Work

- [ ] Larger external benchmark
- [ ] Multi-domain evaluation
- [ ] Confidence calibration
- [ ] Interactive demonstration interface

---

## System Pipeline

```text
Scientific PDFs
      |
      v
Page-level extraction
      |
      v
Stable overlapping chunks
      |
      +-----------------------+
      |                       |
      v                       v
    BM25                Dense retrieval
      |                       |
      +-----------+-----------+
                  |
                  v
       Reciprocal Rank Fusion
                  |
                  v
        Cross-encoder reranking
                  |
                  v
        Retrieval abstention
                  |
          +-------+-------+
          |               |
     insufficient      sufficient
       evidence         evidence
          |               |
          v               v
       Abstain      Citation-grounded
                       generation
                           |
                           v
                    Claim extraction
                           |
                           v
                  Evidence-window NLI
                     verification
                           |
                           v
                  Unsupported-claim
                       filtering
                           |
                           v
                    Verified answer
```

---

## Phase 2 Evaluation

The final Phase 2 pilot evaluation used a **manually audited 11-question benchmark**:

- **9 answerable questions**
- **2 intentionally unanswerable questions**

| Metric | Result |
|---|---:|
| Citation validity | **1.00** |
| Citation grounding | **1.00** |
| Mean claim verification support | **0.829** |
| Mean confidence for answered questions | **0.547** |
| Answerable questions answered | **8 / 9** |
| Unanswerable questions correctly abstained | **2 / 2** |
| Unanswerable questions answered | **0 / 2** |
| Automated tests | **58 passed** |

### Interpretation

**Citation validity = 1.00**  
All citations produced for answered questions referred to valid chunk identifiers available to the system.

**Citation grounding = 1.00**  
Every answered benchmark question cited at least one passage included in the manually audited gold evidence.

**Claim verification support = 0.829**  
Approximately 83% of generated factual claims passed claim-level evidence verification. This metric is intentionally stricter than citation validity.

**Abstention**  
EvidenceGuard answered 8 of 9 answerable questions and correctly abstained on both intentionally unanswerable questions.

**Confidence**  
The current confidence score is an experimental composite signal and should **not** be interpreted as a calibrated probability of correctness.

Detailed Phase 2 report:

[`report/02_generation_verification_results.md`](report/02_generation_verification_results.md)

Machine-readable results:

[`results/generation_eval_results.jsonl`](results/generation_eval_results.jsonl)

---

## Phase 1 Retrieval Results

The original Phase 1 pilot retrieval benchmark produced:

| Method | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.056 | 0.194 | 0.361 | 0.159 |
| Dense | 0.250 | 0.361 | 0.546 | 0.224 |
| Hybrid | 0.259 | 0.343 | 0.556 | 0.227 |
| Rerank | 0.306 | 0.435 | 0.556 | 0.329 |

Cross-encoder reranking produced the strongest ranking quality on the pilot benchmark.

Detailed Phase 1 report:

[`report/01_retrieval_results.md`](report/01_retrieval_results.md)

---

## Benchmark Audit

The original benchmark is preserved unchanged.

During Phase 2 evaluation, manual evidence auditing identified:

- valid alternative evidence absent from some original gold labels;
- ambiguous questions requiring clarification;
- incomplete reference answers;
- intentionally unanswerable questions that should remain unanswerable.

The final Phase 2 evaluation therefore uses:

[`data/evaluation/benchmark_audited.jsonl`](data/evaluation/benchmark_audited.jsonl)

This prevents the system from being penalized for citing correct evidence that was absent from the original annotations.

---

## Verification Design

Claim verification uses a Natural Language Inference cross-encoder.

The verification pipeline includes:

- atomic claim extraction;
- citation-to-evidence mapping;
- multi-sentence evidence windows;
- terminology normalization;
- claim-level entailment checking;
- post-verification filtering.

Regression testing identified cases where abbreviation mismatch or insufficient evidence context caused false negatives. These cases were converted into automated tests.

An experimental NLI-based **reference-alignment diagnostic** was also investigated during development. It was excluded from headline metrics after validation showed unacceptable false-negative behavior on near-identical claim/reference pairs.

---

## Reproducibility

The project includes automated tests covering generation behavior, citation handling, retrieval-based abstention, calibration, terminology normalization, evidence-window construction, claim verification, verification integration, post-verification filtering, and evaluation behavior.

Current validated test status:

```text
58 passed
```

Run the tests with:

```bash
pytest -q
```

---

## Setup

```bash
git clone https://github.com/dhshahin/evidenceguard.git
cd evidenceguard
pip install -r requirements.txt
```

For generation evaluation, provide the Anthropic API key through an environment variable:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Do not commit API keys or credentials.

---

## Reproduce Retrieval Evaluation

Place the source PDFs in:

```text
data/raw/
```

Build the corpus:

```bash
python scripts_build_chunks.py
```

Run retrieval evaluation:

```bash
python run_retrieval_eval.py
```

---

## Reproduce Generation Evaluation

After building the corpus:

```bash
python run_generation_eval.py
```

The final Phase 2 evaluation uses the manually audited benchmark and writes detailed machine-readable results to:

```text
results/generation_eval_results.jsonl
```

---

## Limitations

EvidenceGuard remains a **research prototype**.

The current evaluation is limited by:

- five scientific papers;
- eleven benchmark questions;
- one primary scientific domain;
- manually audited relevance annotations;
- only two intentionally unanswerable questions;
- possible run-to-run variation in LLM generation;
- an imperfect learned NLI verifier;
- an uncalibrated confidence score.

The reported results therefore characterize this prototype and pilot benchmark rather than establish production-level performance across scientific evidence synthesis tasks.

---

## Research Principles

EvidenceGuard is developed around four principles:

1. **Traceability** — factual answers should connect to identifiable source evidence.
2. **Abstention** — insufficient evidence should produce abstention rather than unsupported answers.
3. **Claim-level verification** — evidence support should be checked for individual factual claims.
4. **Transparent evaluation** — retrieval, citations, verification, abstention, and confidence should be measured separately.

---

## Disclaimer

EvidenceGuard is an experimental research project.

It is not intended to provide medical advice, clinical recommendations, or automated scientific conclusions without expert review.
