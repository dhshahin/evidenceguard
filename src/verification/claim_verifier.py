
"""Claim-level verification for citation-grounded generated answers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from src.verification.terminology_normalization import normalize_terminology


NLI_LABELS = (
    "contradiction",
    "entailment",
    "neutral",
)


@dataclass
class Claim:
    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class ClaimVerification:
    claim: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    label: str = "neutral"
    entailment_score: float = 0.0
    supported: bool = False
    best_evidence_chunk_id: str | None = None
    best_evidence_text: str | None = None


@dataclass
class VerificationReport:
    claims: list[ClaimVerification] = field(default_factory=list)
    support_rate: float = 0.0
    supported_claims: int = 0
    unsupported_claims: int = 0
    uncited_claims: int = 0


def _parse_citation_ids(text: str) -> list[str]:
    ids: list[str] = []

    for bracket in re.findall(r"\[([^\]]+)\]", text):
        for piece in bracket.split(","):
            citation = piece.strip()

            if citation and citation not in ids:
                ids.append(citation)

    return ids


def _remove_citations(text: str) -> str:
    cleaned = re.sub(r"\s*\[[^\]]+\]", "", text)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_evidence_windows(
    text: str,
    max_sentences: int = 2,
) -> list[str]:
    """Build overlapping sentence windows from an evidence passage."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1")

    sentences = _split_sentences(text)

    if not sentences:
        return []

    windows: list[str] = []

    largest_window = min(
        max_sentences,
        len(sentences),
    )

    for size in range(1, largest_window + 1):
        for start in range(
            0,
            len(sentences) - size + 1,
        ):
            window = " ".join(
                sentences[start:start + size]
            )

            if window not in windows:
                windows.append(window)

    return windows


def extract_claims(answer: str) -> list[Claim]:
    """Extract atomic claims and associate trailing citations with each claim."""
    answer = answer.strip()

    if not answer:
        return []

    blocks = re.split(
        r"\n\s*\n+",
        answer,
    )

    segments: list[str] = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        block = re.sub(
            r"^\s*(?:[-*]|\d+[.)])\s*",
            "",
            block,
        )

        if (
            not _parse_citation_ids(block)
            and block.endswith(":")
        ):
            continue

        block_segments = re.split(
            r"(?<=\])\s+(?=[A-Z0-9])"
            r"|(?<=[.!?])\s+(?=[A-Z0-9])",
            block,
        )

        segments.extend(
            segment.strip()
            for segment in block_segments
            if segment.strip()
        )

    claims: list[Claim] = []

    for segment in segments:
        claim_text = _remove_citations(
            segment
        )

        if not claim_text:
            continue

        claims.append(
            Claim(
                text=claim_text,
                cited_chunk_ids=_parse_citation_ids(
                    segment
                ),
            )
        )

    return claims


def label_from_scores(
    scores: Sequence[float],
) -> tuple[str, float]:
    """Return the highest-scoring NLI label and its score."""
    if len(scores) != 3:
        raise ValueError(
            "NLI scores must contain exactly three values"
        )

    best_index = max(
        range(len(scores)),
        key=lambda i: scores[i],
    )

    return (
        NLI_LABELS[best_index],
        float(scores[best_index]),
    )


def _softmax(values: Sequence[float]) -> list[float]:
    """Convert model logits to probabilities."""
    if len(values) == 0:
        raise ValueError(
            "cannot compute softmax of an empty sequence"
        )

    maximum = max(
        float(value)
        for value in values
    )

    exponentials = [
        math.exp(float(value) - maximum)
        for value in values
    ]

    denominator = sum(exponentials)

    return [
        value / denominator
        for value in exponentials
    ]


def _select_best_prediction(
    probabilities: list[list[float]],
) -> int:
    labels = [
        label_from_scores(row)[0]
        for row in probabilities
    ]

    entailment_indices = [
        i
        for i, label in enumerate(labels)
        if label == "entailment"
    ]

    if entailment_indices:
        return max(
            entailment_indices,
            key=lambda i: probabilities[i][1],
        )

    contradiction_indices = [
        i
        for i, label in enumerate(labels)
        if label == "contradiction"
    ]

    if contradiction_indices:
        return max(
            contradiction_indices,
            key=lambda i: probabilities[i][0],
        )

    return max(
        range(len(probabilities)),
        key=lambda i: probabilities[i][2],
    )


def verify_claim(
    claim: Claim,
    passages: Mapping[str, str],
    model,
    entailment_threshold: float = 0.5,
    evidence_window_sentences: int | None = None,
    terminology_aliases: Mapping[str, str] | None = None,
) -> ClaimVerification:
    """Verify one claim against its cited evidence."""
    if not claim.cited_chunk_ids:
        return ClaimVerification(
            claim=claim.text,
            cited_chunk_ids=[],
            label="uncited",
            entailment_score=0.0,
            supported=False,
            best_evidence_chunk_id=None,
            best_evidence_text=None,
        )

    available_ids = [
        chunk_id
        for chunk_id in claim.cited_chunk_ids
        if chunk_id in passages
    ]

    if not available_ids:
        return ClaimVerification(
            claim=claim.text,
            cited_chunk_ids=claim.cited_chunk_ids,
            label="missing_evidence",
            entailment_score=0.0,
            supported=False,
            best_evidence_chunk_id=None,
            best_evidence_text=None,
        )

    pairs: list[list[str]] = []
    evidence_metadata: list[tuple[str, str]] = []

    for chunk_id in available_ids:
        passage_text = passages[chunk_id]

        if evidence_window_sentences is None:
            windows = [passage_text]
        else:
            windows = build_evidence_windows(
                passage_text,
                max_sentences=evidence_window_sentences,
            )

            if not windows:
                windows = [passage_text]

        for window in windows:
            normalized_window = normalize_terminology(
                window,
                terminology_aliases,
            )

            normalized_claim = normalize_terminology(
                claim.text,
                terminology_aliases,
            )

            pairs.append([
                normalized_window,
                normalized_claim,
            ])

            evidence_metadata.append(
                (chunk_id, window)
            )

    raw_outputs = model.predict(pairs)

    probabilities = [
        _softmax(row)
        for row in raw_outputs
    ]

    if not probabilities:
        return ClaimVerification(
            claim=claim.text,
            cited_chunk_ids=claim.cited_chunk_ids,
            label="missing_evidence",
            entailment_score=0.0,
            supported=False,
            best_evidence_chunk_id=None,
            best_evidence_text=None,
        )

    best_index = _select_best_prediction(
        probabilities
    )

    best_probabilities = probabilities[
        best_index
    ]

    best_chunk_id, best_evidence_text = (
        evidence_metadata[best_index]
    )

    label, _ = label_from_scores(
        best_probabilities
    )

    entailment_score = float(
        best_probabilities[1]
    )

    supported = (
        label == "entailment"
        and entailment_score
        >= entailment_threshold
    )

    return ClaimVerification(
        claim=claim.text,
        cited_chunk_ids=claim.cited_chunk_ids,
        label=label,
        entailment_score=entailment_score,
        supported=supported,
        best_evidence_chunk_id=best_chunk_id,
        best_evidence_text=best_evidence_text,
    )


def verification_support_rate(
    claims: list[ClaimVerification],
) -> float:
    """Return the fraction of claims classified as supported."""
    if not claims:
        return 0.0

    supported = sum(
        1
        for claim in claims
        if claim.supported
    )

    return supported / len(claims)


def verify_answer(
    answer: str,
    passages: Mapping[str, str],
    model,
    entailment_threshold: float = 0.5,
    evidence_window_sentences: int | None = None,
    terminology_aliases: Mapping[str, str] | None = None,
) -> VerificationReport:
    """Verify every claim in a generated answer."""
    extracted = extract_claims(answer)

    verified = [
        verify_claim(
            claim,
            passages,
            model,
            entailment_threshold=entailment_threshold,
            evidence_window_sentences=evidence_window_sentences,
            terminology_aliases=terminology_aliases,
        )
        for claim in extracted
    ]

    supported_claims = sum(
        1
        for claim in verified
        if claim.supported
    )

    unsupported_claims = (
        len(verified) - supported_claims
    )

    uncited_claims = sum(
        1
        for claim in verified
        if claim.label == "uncited"
    )

    return VerificationReport(
        claims=verified,
        support_rate=verification_support_rate(
            verified
        ),
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        uncited_claims=uncited_claims,
    )


def render_verified_answer(
    report: VerificationReport,
) -> str:
    """Render only claims that passed evidence verification."""
    rendered_claims: list[str] = []

    for item in report.claims:
        if not item.supported:
            continue

        claim_text = item.claim.strip()

        if not claim_text:
            continue

        citations = " ".join(
            f"[{chunk_id}]"
            for chunk_id in item.cited_chunk_ids
        )

        if citations:
            rendered_claims.append(
                f"{claim_text} {citations}"
            )
        else:
            rendered_claims.append(
                claim_text
            )

    return "\n\n".join(rendered_claims)


def build_verified_final_answer(
    report: VerificationReport,
) -> tuple[str, bool]:
    """Build the final answer after claim-level verification."""
    from src.generation.generator import ABSTAIN_MARKER

    verified_text = render_verified_answer(
        report
    ).strip()

    if not verified_text:
        return ABSTAIN_MARKER, True

    return verified_text, False

