"""Confidence scoring for generated answers.

A deliberately simple, explainable score rather than a black box. Two
signals combine:

  * retrieval strength  — how strong the top evidence was (normalised),
  * citation coverage   — how many distinct offered passages the answer
                          actually cited (an answer grounded in several
                          passages is more trustworthy than one leaning on
                          a single weak hit).

The score is a transparent weighted blend in [0, 1]. It is NOT a
calibrated probability; it is a relative trust signal, and is documented
as such. Keeping it interpretable is the point — every input can be shown
to a user alongside the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.generation.generator import GeneratedAnswer
from src.retrieval.base import RetrievalResult


@dataclass
class ConfidenceReport:
    score: float                 # overall confidence in [0, 1]
    retrieval_strength: float    # normalised top-score component
    citation_coverage: float     # fraction of offered passages cited
    explanation: str


def _normalise(score: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (score - lo) / (hi - lo)))


def score_confidence(
    answer: GeneratedAnswer,
    passages: list[RetrievalResult],
    score_lo: float,
    score_hi: float,
    w_retrieval: float = 0.5,
    w_citation: float = 0.5,
) -> ConfidenceReport:
    """Blend retrieval strength and citation coverage into a trust signal.

    score_lo / score_hi are the observed min/max top-scores for this
    retriever (from the benchmark), used to normalise the raw score onto
    [0, 1] so the number is comparable across questions.
    """
    if answer.abstained:
        return ConfidenceReport(
            score=0.0, retrieval_strength=0.0, citation_coverage=0.0,
            explanation="Abstained — no answer produced.")

    top = passages[0].score if passages else 0.0
    retrieval_strength = _normalise(top, score_lo, score_hi)

    n_offered = len(answer.passages_used)
    citation_coverage = (len(answer.cited_chunk_ids) / n_offered) if n_offered else 0.0

    score = w_retrieval * retrieval_strength + w_citation * citation_coverage
    explanation = (
        f"retrieval_strength={retrieval_strength:.2f} "
        f"(top score {top:.2f} in [{score_lo:.2f}, {score_hi:.2f}]), "
        f"citation_coverage={citation_coverage:.2f} "
        f"({len(answer.cited_chunk_ids)}/{n_offered} passages cited)")
    return ConfidenceReport(
        score=score, retrieval_strength=retrieval_strength,
        citation_coverage=citation_coverage, explanation=explanation)
