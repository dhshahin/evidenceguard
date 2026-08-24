
import json
from pathlib import Path

import pytest
import torch
from sentence_transformers import CrossEncoder

from src.config import NLI_MODEL
from src.verification.claim_verifier import (
    Claim,
    verify_claim,
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


@pytest.fixture(scope="module")
def chunks():
    path = Path(
        "data/processed/chunks.jsonl"
    )

    return {
        item["chunk_id"]: item["text"]
        for item in (
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
    }


def test_dl_abbreviation_is_verified_as_deep_learning(
    real_nli_model,
    chunks,
):
    chunk_id = "fonc-14-1343627::p1::c0"

    result = verify_claim(
        claim=Claim(
            text=(
                "The review investigated the application "
                "of deep learning to digital mammography."
            ),
            cited_chunk_ids=[chunk_id],
        ),
        passages={
            chunk_id: chunks[chunk_id]
        },
        model=real_nli_model,
        entailment_threshold=0.5,
        evidence_window_sentences=2,
        terminology_aliases={
            "DL": "deep learning",
        },
    )

    assert result.supported is True


def test_top_mortality_predictor_gets_sufficient_context(
    real_nli_model,
    chunks,
):
    chunk_id = (
        "cancers-16-03799::p14::c0"
    )

    result = verify_claim(
        claim=Claim(
            text=(
                "Occurrence in other organs was identified "
                "as a top mortality predictor."
            ),
            cited_chunk_ids=[chunk_id],
        ),
        passages={
            chunk_id: chunks[chunk_id]
        },
        model=real_nli_model,
        entailment_threshold=0.5,
        evidence_window_sentences=3,
    )

    assert result.supported is True
