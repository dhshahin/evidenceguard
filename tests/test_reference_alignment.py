
import pytest
import torch
from sentence_transformers import CrossEncoder

from src.config import NLI_MODEL
from src.evaluation.reference_alignment import (
    align_claim_to_reference,
)


@pytest.fixture(scope="module")
def real_nli_model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    return CrossEncoder(
        NLI_MODEL,
        device=device,
    )


def test_reference_alignment_accepts_requested_top_predictor(
    real_nli_model,
):
    reference = (
        "The top four mortality predictors were occurrence in "
        "other organs, N stage, age at diagnosis, and radiation "
        "treatment for curative purposes."
    )

    result = align_claim_to_reference(
        claim_text=(
            "N stage was identified as a top mortality predictor."
        ),
        reference_answer=reference,
        model=real_nli_model,
    )

    assert result.aligned is True


def test_reference_alignment_rejects_supported_but_out_of_scope_claim(
    real_nli_model,
):
    reference = (
        "The top four mortality predictors were occurrence in "
        "other organs, N stage, age at diagnosis, and radiation "
        "treatment for curative purposes."
    )

    result = align_claim_to_reference(
        claim_text=(
            "T stage was identified as a key mortality predictor."
        ),
        reference_answer=reference,
        model=real_nli_model,
    )

    assert result.aligned is False
