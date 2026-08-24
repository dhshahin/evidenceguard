
from src.verification.claim_verifier import (
    Claim,
    verify_claim,
    verify_answer,
)


class FakeNLIModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict(self, pairs):
        return self.outputs[:len(pairs)]


def test_verify_claim_supported():
    claim = Claim(
        text="The model achieved high predictive performance.",
        cited_chunk_ids=["paper::p1::c0"],
    )

    passages = {
        "paper::p1::c0": (
            "The proposed model achieved high predictive performance."
        )
    }

    model = FakeNLIModel([
        [0.1, 3.0, 0.2]
    ])

    result = verify_claim(
        claim,
        passages,
        model,
        entailment_threshold=0.5,
    )

    assert result.supported is True
    assert result.label == "entailment"
    assert result.best_evidence_chunk_id == "paper::p1::c0"


def test_verify_claim_contradicted():
    claim = Claim(
        text="The study used 500 participants.",
        cited_chunk_ids=["paper::p2::c0"],
    )

    passages = {
        "paper::p2::c0": "The study included only 100 participants."
    }

    model = FakeNLIModel([
        [4.0, 0.1, 0.2]
    ])

    result = verify_claim(
        claim,
        passages,
        model,
        entailment_threshold=0.5,
    )

    assert result.supported is False
    assert result.label == "contradiction"


def test_verify_claim_selects_best_cited_evidence():
    claim = Claim(
        text="Random forest was the strongest model.",
        cited_chunk_ids=[
            "paper::p1::c0",
            "paper::p2::c0",
        ],
    )

    passages = {
        "paper::p1::c0": "Several models were evaluated.",
        "paper::p2::c0": "Random forest had the best performance.",
    }

    model = FakeNLIModel([
        [0.2, 0.1, 2.0],
        [0.1, 4.0, 0.1],
    ])

    result = verify_claim(
        claim,
        passages,
        model,
        entailment_threshold=0.5,
    )

    assert result.supported is True
    assert result.best_evidence_chunk_id == "paper::p2::c0"


def test_verify_uncited_claim():
    claim = Claim(
        text="The study was externally validated.",
        cited_chunk_ids=[],
    )

    result = verify_claim(
        claim,
        {},
        FakeNLIModel([]),
    )

    assert result.supported is False
    assert result.label == "uncited"


def test_verify_answer_builds_report():
    answer = (
        "The model performed well [paper::p1::c0]. "
        "External validation was performed."
    )

    passages = {
        "paper::p1::c0": "The model demonstrated strong performance."
    }

    model = FakeNLIModel([
        [0.1, 3.0, 0.2]
    ])

    report = verify_answer(
        answer,
        passages,
        model,
        entailment_threshold=0.5,
    )

    assert len(report.claims) == 2
    assert report.supported_claims == 1
    assert report.unsupported_claims == 1
    assert report.uncited_claims == 1
    assert report.support_rate == 0.5
