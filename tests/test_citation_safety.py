
import src.generation.generator as gen


def test_raw_citation_extractor_exists():
    """We must retain ALL model citations, including fabricated ones."""
    assert hasattr(gen, "_extract_all_cited_ids"), (
        "Generator currently has no way to preserve raw/fabricated citations "
        "for auditing."
    )


def test_raw_citation_extractor_keeps_invalid_ids():
    """A fabricated citation must remain visible to evaluation."""
    if not hasattr(gen, "_extract_all_cited_ids"):
        return

    text = (
        "Supported claim [paper::p1::c0]. "
        "Fabricated claim [fake-paper::p99::c9]."
    )

    ids = gen._extract_all_cited_ids(text)

    assert ids == [
        "paper::p1::c0",
        "fake-paper::p99::c9",
    ]


def test_generated_answer_tracks_raw_citations():
    """GeneratedAnswer must preserve citations before validity filtering."""
    fields = gen.GeneratedAnswer.__dataclass_fields__

    assert "raw_cited_chunk_ids" in fields, (
        "GeneratedAnswer currently stores only filtered valid citations, "
        "so fabricated citations can disappear before evaluation."
    )


def test_fabricated_citation_reduces_validity():
    """One valid and one fabricated citation should yield validity of 0.5."""
    from src.evaluation.generation_metrics import citation_validity

    ans = gen.GeneratedAnswer(
        question="q",
        answer=(
            "Supported [paper::p1::c0]. "
            "Unsupported [fake-paper::p99::c9]."
        ),
        cited_chunk_ids=["paper::p1::c0"],
        raw_cited_chunk_ids=[
            "paper::p1::c0",
            "fake-paper::p99::c9",
        ],
        passages_used=["paper::p1::c0"],
    )

    assert citation_validity(ans) == 0.5
