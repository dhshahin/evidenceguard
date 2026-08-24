# EvidenceGuard Phase 2 Evaluation

## Overview

This report summarizes the final pilot evaluation of EvidenceGuard Phase 2:
citation-grounded scientific generation, abstention, claim-level verification,
and confidence estimation.

The final evaluation uses the manually audited 11-question benchmark:

- 9 answerable questions
- 2 intentionally unanswerable questions

## Headline Results

| Metric | Result |
|---|---:|
| Citation validity | 1.000 |
| Citation grounding | 1.000 |
| Mean claim verification support | 0.829 |
| Mean confidence for answered questions | 0.547 |
| Answerable questions answered | 8/9 |
| Unanswerable questions correctly abstained | 2/2 |
| Automated tests | 58 passed |

## Interpretation

Citation validity of 1.00 means that citations produced for
answered questions referred to valid chunk identifiers.

Citation grounding of 1.00 means that every answered
question cited at least one chunk included in the manually audited gold
evidence.

Mean claim-level verification support was 0.829.
Verification is deliberately stricter than citation validity because a real
citation does not necessarily entail every generated claim.

EvidenceGuard answered 8 of 9 answerable
questions and correctly abstained on all 2 intentionally
unanswerable questions.

The current confidence score is an experimental composite signal and should
not be interpreted as a calibrated probability of correctness.

## Benchmark Audit

The original benchmark remains preserved.

The final evaluation uses:

`data/evaluation/benchmark_audited.jsonl`

Manual auditing added valid alternative evidence, corrected incomplete
reference answers, clarified ambiguous questions, and preserved intentionally
unanswerable questions.

## Verification Validation

Development included regression testing for:

- citation safety;
- atomic claim generation;
- terminology normalization;
- evidence-window construction;
- Natural Language Inference verification;
- abstention;
- post-verification filtering;
- evaluation integration.

An experimental NLI-based reference-alignment diagnostic was investigated but
excluded from headline metrics after manual validation showed unacceptable
false-negative behavior.

## Limitations

This remains a pilot research evaluation.

Current limitations include:

- five scientific papers;
- eleven benchmark questions;
- one primary scientific domain;
- manually audited relevance annotations;
- only two intentionally unanswerable questions;
- possible run-to-run variation in LLM generation;
- an imperfect learned NLI verifier;
- an uncalibrated confidence score.

The reported results characterize this prototype and benchmark and should not
be interpreted as production-level general performance.

## Reproducibility

At the time of this evaluation:

**58 automated tests passed.**

Detailed machine-readable results:

`results/generation_eval_results.jsonl`

Audited benchmark:

`data/evaluation/benchmark_audited.jsonl`
