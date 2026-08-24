
import pytest

from src.verification.claim_verifier import (
    extract_claims,
    label_from_scores,
    ClaimVerification,
    VerificationReport,
    verification_support_rate,
)


def test_extract_single_claim_with_citation():
    answer = (
        "The model achieved an AUC of 0.91 "
        "[paper::p4::c1]."
    )

    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0].text == "The model achieved an AUC of 0.91."
    assert claims[0].cited_chunk_ids == ["paper::p4::c1"]


def test_extract_multiple_claims():
    answer = (
        "The dataset included 500 patients [paper::p2::c0]. "
        "Random forest performed best "
        "[paper::p5::c2, paper::p5::c3]."
    )

    claims = extract_claims(answer)

    assert len(claims) == 2

    assert claims[0].cited_chunk_ids == [
        "paper::p2::c0"
    ]

    assert claims[1].cited_chunk_ids == [
        "paper::p5::c2",
        "paper::p5::c3",
    ]


def test_uncited_claim_is_preserved():
    answer = "The method was externally validated."

    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0].text == "The method was externally validated."
    assert claims[0].cited_chunk_ids == []


def test_nli_label_mapping_entailment():
    label, probability = label_from_scores(
        [0.1, 0.8, 0.1]
    )

    assert label == "entailment"
    assert probability == pytest.approx(0.8)


def test_nli_label_mapping_contradiction():
    label, probability = label_from_scores(
        [0.9, 0.05, 0.05]
    )

    assert label == "contradiction"
    assert probability == pytest.approx(0.9)


def test_nli_label_mapping_neutral():
    label, probability = label_from_scores(
        [0.1, 0.2, 0.7]
    )

    assert label == "neutral"
    assert probability == pytest.approx(0.7)


def test_support_rate():
    claims = [
        ClaimVerification(
            claim="Claim one.",
            cited_chunk_ids=["p1"],
            label="entailment",
            entailment_score=0.90,
            supported=True,
        ),
        ClaimVerification(
            claim="Claim two.",
            cited_chunk_ids=["p2"],
            label="neutral",
            entailment_score=0.20,
            supported=False,
        ),
    ]

    assert verification_support_rate(claims) == 0.5


def test_empty_verification_report():
    report = VerificationReport(
        claims=[],
        support_rate=0.0,
        supported_claims=0,
        unsupported_claims=0,
        uncited_claims=0,
    )

    assert report.support_rate == 0.0
    assert report.supported_claims == 0


def test_softmax_accepts_numpy_array():
    import numpy as np
    from src.verification.claim_verifier import _softmax

    values = np.array([0.1, 2.0, 0.3])

    probabilities = _softmax(values)

    assert len(probabilities) == 3
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[1] > probabilities[0]
    assert probabilities[1] > probabilities[2]



def test_extract_claims_splits_sentences_with_trailing_citations():
    answer = (
        "ER status was identified as a key predictor. "
        "[paper::p1::c0] "
        "Tumor size was identified as a key predictor. "
        "[paper::p1::c0]"
    )

    claims = extract_claims(answer)

    assert len(claims) == 2

    assert claims[0].text == (
        "ER status was identified as a key predictor."
    )
    assert claims[0].cited_chunk_ids == [
        "paper::p1::c0"
    ]

    assert claims[1].text == (
        "Tumor size was identified as a key predictor."
    )
    assert claims[1].cited_chunk_ids == [
        "paper::p1::c0"
    ]


def test_extract_claims_handles_multiple_citations_before_next_sentence():
    answer = (
        "The feature was predictive. "
        "[paper::p1::c0] [paper::p2::c1] "
        "The model achieved high performance. "
        "[paper::p3::c0]"
    )

    claims = extract_claims(answer)

    assert len(claims) == 2

    assert claims[0].cited_chunk_ids == [
        "paper::p1::c0",
        "paper::p2::c1",
    ]

    assert claims[1].cited_chunk_ids == [
        "paper::p3::c0",
    ]
