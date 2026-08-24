"""Reference-answer alignment evaluation.

This module is used only during benchmark evaluation.
It is not part of the production generation pipeline because
reference answers are unavailable at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.verification.claim_verifier import (
    Claim,
    verify_claim,
)


@dataclass
class ReferenceAlignment:
    claim: str
    aligned: bool
    label: str
    entailment_score: float


def align_claim_to_reference(
    claim_text: str,
    reference_answer: str,
    model,
    entailment_threshold: float = 0.5,
) -> ReferenceAlignment:
    """Check whether a generated claim is supported by the reference answer."""

    result = verify_claim(
        claim=Claim(
            text=claim_text,
            cited_chunk_ids=["reference"],
        ),
        passages={
            "reference": reference_answer,
        },
        model=model,
        entailment_threshold=entailment_threshold,
        evidence_window_sentences=None,
    )

    return ReferenceAlignment(
        claim=claim_text,
        aligned=result.supported,
        label=result.label,
        entailment_score=result.entailment_score,
    )
