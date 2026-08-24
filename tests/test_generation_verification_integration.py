
from types import SimpleNamespace

from run_generation_eval import verify_generated_answer
from src.generation.generator import (
    GeneratedAnswer,
    ABSTAIN_MARKER,
)


class TerminologyAwareFakeNLI:
    def predict(self, pairs):
        outputs = []

        for premise, hypothesis in pairs:
            premise = premise.lower()
            hypothesis = hypothesis.lower()

            if (
                "lymphovascular invasion" in premise
                and "lymphovascular invasion" in hypothesis
            ):
                outputs.append([0.1, 4.0, 0.1])
            else:
                outputs.append([0.1, 0.2, 4.0])

        return outputs


def test_verify_generated_answer_returns_report_for_answered_question():
    chunk_id = "paper::p1::c0"

    results = [
        SimpleNamespace(
            chunk_id=chunk_id,
            text=(
                "The selected clinical predictors included LVI, "
                "age at diagnosis, and tumor size."
            ),
        )
    ]

    answer = GeneratedAnswer(
        question="Which predictors were selected?",
        answer=(
            "Lymphovascular invasion was identified as a predictor "
            f"[{chunk_id}]."
        ),
        cited_chunk_ids=[chunk_id],
        raw_cited_chunk_ids=[chunk_id],
        abstained=False,
        passages_used=[chunk_id],
    )

    report = verify_generated_answer(
        answer,
        results,
        TerminologyAwareFakeNLI(),
    )

    assert report is not None
    assert report.supported_claims == 1
    assert report.unsupported_claims == 0
    assert report.support_rate == 1.0

    verified = report.claims[0]

    assert verified.supported is True
    assert verified.label == "entailment"
    assert "LVI" in verified.best_evidence_text


def test_verify_generated_answer_skips_abstained_answer():
    answer = GeneratedAnswer(
        question="Unanswerable question",
        answer=ABSTAIN_MARKER,
        abstained=True,
        passages_used=[],
    )

    report = verify_generated_answer(
        answer,
        [],
        TerminologyAwareFakeNLI(),
    )

    assert report is None


def test_verify_generated_answer_marks_fabricated_citation_unsupported():
    answer = GeneratedAnswer(
        question="Which predictor was selected?",
        answer=(
            "Lymphovascular invasion was identified as a predictor "
            "[fabricated::p1::c0]."
        ),
        cited_chunk_ids=[],
        raw_cited_chunk_ids=[
            "fabricated::p1::c0",
        ],
        abstained=False,
        passages_used=[
            "paper::p1::c0",
        ],
    )

    results = [
        SimpleNamespace(
            chunk_id="paper::p1::c0",
            text="The selected clinical predictor was LVI.",
        )
    ]

    report = verify_generated_answer(
        answer,
        results,
        TerminologyAwareFakeNLI(),
    )

    assert report is not None
    assert report.supported_claims == 0
    assert report.unsupported_claims == 1
    assert report.support_rate == 0.0
    assert report.claims[0].label == "missing_evidence"
