
from src.verification.claim_verifier import (
    ClaimVerification,
    build_evidence_windows,
    Claim,
    verify_claim,
)


class FakeNLIModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict(self, pairs):
        return self.outputs[:len(pairs)]


def test_build_evidence_windows_single_sentences():
    text = (
        "Sentence one contains information. "
        "Sentence two contains more information. "
        "Sentence three contains the result."
    )

    windows = build_evidence_windows(
        text,
        max_sentences=1,
    )

    assert windows == [
        "Sentence one contains information.",
        "Sentence two contains more information.",
        "Sentence three contains the result.",
    ]


def test_build_evidence_windows_two_sentence_overlap():
    text = (
        "Sentence one contains information. "
        "Sentence two contains more information. "
        "Sentence three contains the result."
    )

    windows = build_evidence_windows(
        text,
        max_sentences=2,
    )

    assert "Sentence one contains information. Sentence two contains more information." in windows
    assert "Sentence two contains more information. Sentence three contains the result." in windows


def test_verify_claim_uses_focused_window():
    claim = Claim(
        text="ER status was identified as a key predictor.",
        cited_chunk_ids=["paper::p1::c0"],
    )

    passages = {
        "paper::p1::c0": (
            "The introduction discusses breast cancer. "
            "Several clinical variables were initially considered. "
            "ER status was identified as a key predictor. "
            "The ensemble model was evaluated using AUC."
        )
    }

    model = FakeNLIModel([
        [0.1, 0.2, 3.0],
        [0.1, 0.2, 3.0],
        [0.1, 4.0, 0.1],
        [0.1, 0.2, 3.0],
    ])

    result = verify_claim(
        claim,
        passages,
        model,
        entailment_threshold=0.5,
        evidence_window_sentences=1,
    )

    assert result.supported is True
    assert result.label == "entailment"
    assert "ER status was identified as a key predictor" in result.best_evidence_text


def test_verification_retains_best_evidence_text():
    fields = ClaimVerification.__dataclass_fields__

    assert "best_evidence_text" in fields
