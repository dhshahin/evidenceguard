
from src.verification.claim_verifier import (
    ClaimVerification,
    VerificationReport,
    render_verified_answer,
)


def test_render_verified_answer_keeps_supported_claims():
    report = VerificationReport(
        claims=[
            ClaimVerification(
                claim="ER status was identified as a predictor.",
                cited_chunk_ids=["paper::p1::c0"],
                label="entailment",
                entailment_score=0.99,
                supported=True,
                best_evidence_chunk_id="paper::p1::c0",
                best_evidence_text="ER status was a predictor.",
            ),
            ClaimVerification(
                claim="ER status was the second most important predictor.",
                cited_chunk_ids=["paper::p1::c0"],
                label="contradiction",
                entailment_score=0.01,
                supported=False,
            ),
        ],
        support_rate=0.5,
        supported_claims=1,
        unsupported_claims=1,
        uncited_claims=0,
    )

    rendered = render_verified_answer(report)

    assert (
        "ER status was identified as a predictor. "
        "[paper::p1::c0]"
    ) in rendered

    assert "second most important" not in rendered


def test_render_verified_answer_preserves_multiple_citations():
    report = VerificationReport(
        claims=[
            ClaimVerification(
                claim="The result was supported across two passages.",
                cited_chunk_ids=[
                    "paper::p1::c0",
                    "paper::p2::c1",
                ],
                label="entailment",
                entailment_score=0.98,
                supported=True,
            )
        ],
        support_rate=1.0,
        supported_claims=1,
        unsupported_claims=0,
        uncited_claims=0,
    )

    rendered = render_verified_answer(report)

    assert (
        "[paper::p1::c0] [paper::p2::c1]"
        in rendered
    )


def test_render_verified_answer_returns_empty_when_nothing_supported():
    report = VerificationReport(
        claims=[
            ClaimVerification(
                claim="Unsupported statement.",
                cited_chunk_ids=["paper::p1::c0"],
                label="neutral",
                entailment_score=0.01,
                supported=False,
            )
        ],
        support_rate=0.0,
        supported_claims=0,
        unsupported_claims=1,
        uncited_claims=0,
    )

    assert render_verified_answer(report) == ""


def test_verified_final_answer_abstains_when_nothing_supported():
    from src.verification.claim_verifier import (
        build_verified_final_answer,
    )
    from src.generation.generator import (
        ABSTAIN_MARKER,
    )

    report = VerificationReport(
        claims=[
            ClaimVerification(
                claim="Unsupported statement.",
                cited_chunk_ids=["paper::p1::c0"],
                label="neutral",
                entailment_score=0.01,
                supported=False,
            )
        ],
        support_rate=0.0,
        supported_claims=0,
        unsupported_claims=1,
        uncited_claims=0,
    )

    final_answer, abstained = (
        build_verified_final_answer(
            report
        )
    )

    assert final_answer == ABSTAIN_MARKER
    assert abstained is True


def test_verified_final_answer_keeps_supported_content():
    from src.verification.claim_verifier import (
        build_verified_final_answer,
    )

    report = VerificationReport(
        claims=[
            ClaimVerification(
                claim="Tumor size was identified as a predictor.",
                cited_chunk_ids=["paper::p1::c0"],
                label="entailment",
                entailment_score=0.99,
                supported=True,
            )
        ],
        support_rate=1.0,
        supported_claims=1,
        unsupported_claims=0,
        uncited_claims=0,
    )

    final_answer, abstained = (
        build_verified_final_answer(
            report
        )
    )

    assert (
        "Tumor size was identified as a predictor."
        in final_answer
    )
    assert abstained is False
