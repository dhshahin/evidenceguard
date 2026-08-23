"""Abstention: decide when NOT to answer.

Two complementary gates:

  1. Retrieval gate (this module): before calling the LLM, check whether
     retrieval surfaced anything strong enough to be worth answering from.
     Cheap, and catches the "nothing relevant found" case.

  2. Generation gate (in generator.py): the model itself replies
     INSUFFICIENT_EVIDENCE when the passages don't actually answer the
     question. Catches "passages retrieved but off-topic".

Using both is more robust than either alone: the retrieval gate is fast and
blocks obvious no-evidence cases without spending an API call; the
generation gate handles subtler cases where passages look plausible but
don't contain the answer.

The retriever's top score is used as the evidence-strength signal. Because
different retrievers use different score scales, the threshold is
retriever-specific and set from the calibration questions, not hard-coded
blindly.
"""
from __future__ import annotations

from src.retrieval.base import RetrievalResult


def top_score(passages: list[RetrievalResult]) -> float:
    """Evidence-strength signal: the best passage's score (0 if none)."""
    return passages[0].score if passages else 0.0


def should_abstain_on_retrieval(
    passages: list[RetrievalResult],
    threshold: float,
) -> bool:
    """Return True if retrieval evidence is too weak to attempt an answer."""
    if not passages:
        return True
    return top_score(passages) < threshold


def suggest_threshold(
    answerable_top_scores: list[float],
    abstain_top_scores: list[float],
) -> float:
    """Pick a threshold separating answerable from abstention questions.

    Simple, defensible choice: the midpoint between the mean top-score of
    answerable questions and the mean top-score of abstention questions.
    With a tiny benchmark this is a heuristic, not a tuned hyperparameter —
    documented as such.
    """
    if not answerable_top_scores or not abstain_top_scores:
        raise ValueError("need both answerable and abstention scores")
    a = sum(answerable_top_scores) / len(answerable_top_scores)
    b = sum(abstain_top_scores) / len(abstain_top_scores)
    return (a + b) / 2.0
