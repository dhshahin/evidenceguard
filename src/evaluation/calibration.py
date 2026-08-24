
"""Calibration utilities for abstention and confidence scoring."""

from __future__ import annotations

from src.generation.abstention import suggest_threshold


def leave_one_out_calibration(
    scores: list[float],
    labels: list[bool],
) -> list[tuple[float, float, float]]:
    """Build leakage-free calibration values for each evaluation item.

    For every item, the current observation is excluded before computing:
      - abstention threshold
      - minimum calibration score
      - maximum calibration score

    labels=True denotes an answerable question.
    labels=False denotes an abstention question.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    if len(scores) < 3:
        raise ValueError("at least three observations are required")

    folds: list[tuple[float, float, float]] = []

    for held_out in range(len(scores)):
        train_scores = [
            score
            for i, score in enumerate(scores)
            if i != held_out
        ]

        train_labels = [
            label
            for i, label in enumerate(labels)
            if i != held_out
        ]

        answerable_scores = [
            score
            for score, label in zip(train_scores, train_labels)
            if label
        ]

        abstention_scores = [
            score
            for score, label in zip(train_scores, train_labels)
            if not label
        ]

        if not answerable_scores or not abstention_scores:
            raise ValueError(
                "each leave-one-out fold must contain both "
                "answerable and abstention examples"
            )

        threshold = suggest_threshold(
            answerable_scores,
            abstention_scores,
        )

        score_lo = min(train_scores)
        score_hi = max(train_scores)

        folds.append(
            (threshold, score_lo, score_hi)
        )

    return folds
