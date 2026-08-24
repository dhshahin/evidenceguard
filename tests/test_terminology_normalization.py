
from src.verification.terminology_normalization import (
    normalize_terminology,
)
from src.verification.claim_verifier import (
    Claim,
    verify_claim,
)


def test_normalize_known_abbreviation():
    aliases = {
        "LVI": "lymphovascular invasion",
    }

    text = "ER status, LVI, Ki-67 index"

    normalized = normalize_terminology(
        text,
        aliases,
    )

    assert normalized == (
        "ER status, lymphovascular invasion, Ki-67 index"
    )


def test_normalization_is_case_insensitive():
    aliases = {
        "LVI": "lymphovascular invasion",
    }

    text = "lvi was recorded."

    normalized = normalize_terminology(
        text,
        aliases,
    )

    assert normalized == (
        "lymphovascular invasion was recorded."
    )


def test_normalization_respects_token_boundaries():
    aliases = {
        "ER": "estrogen receptor",
    }

    text = "ER status was recorded together with HER2."

    normalized = normalize_terminology(
        text,
        aliases,
    )

    assert normalized == (
        "estrogen receptor status was recorded together with HER2."
    )


class RecordingNLIModel:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs = pairs

        outputs = []

        for premise, hypothesis in pairs:
            if "lymphovascular invasion" in premise.lower():
                outputs.append([0.1, 4.0, 0.1])
            else:
                outputs.append([0.1, 0.2, 4.0])

        return outputs


def test_verify_claim_normalizes_nli_input_but_preserves_original_evidence():
    evidence_id = "paper::p1::c0"

    passages = {
        evidence_id: (
            "Six clinical predictors were selected. "
            "The predictors included ER status, LVI, Ki-67 index, "
            "age at diagnosis, and tumor size."
        )
    }

    claim = Claim(
        text=(
            "Lymphovascular invasion was identified "
            "as a clinical predictor."
        ),
        cited_chunk_ids=[evidence_id],
    )

    model = RecordingNLIModel()

    result = verify_claim(
        claim=claim,
        passages=passages,
        model=model,
        entailment_threshold=0.5,
        evidence_window_sentences=2,
        terminology_aliases={
            "LVI": "lymphovascular invasion",
        },
    )

    normalized_premises = [
        premise.lower()
        for premise, _ in model.pairs
    ]

    assert any(
        "lymphovascular invasion" in premise
        for premise in normalized_premises
    )

    assert result.supported is True

    assert "LVI" in result.best_evidence_text
