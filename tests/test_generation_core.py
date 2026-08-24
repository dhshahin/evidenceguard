
import pytest

from src.generation.generator import (
    GeneratedAnswer,
    _extract_cited_ids,
    ABSTAIN_MARKER,
)
from src.generation.abstention import (
    should_abstain_on_retrieval,
    suggest_threshold,
)
from src.confidence.confidence import score_confidence
from src.evaluation.generation_metrics import (
    citation_validity,
    citation_grounding,
    AbstentionCounts,
    tally_abstention,
)
from src.retrieval.base import RetrievalResult


def passage(chunk_id="paper::p1::c0", score=5.0):
    return RetrievalResult(
        chunk_id=chunk_id,
        text="Scientific evidence passage.",
        score=score,
        paper_id="paper",
        title="Paper",
        page_number=1,
        rank=1,
    )


def test_extract_valid_citations():
    valid = {"paper::p1::c0", "paper::p2::c1"}

    text = (
        "Claim one [paper::p1::c0]. "
        "Claim two [paper::p2::c1]."
    )

    ids = _extract_cited_ids(text, valid)

    assert ids == [
        "paper::p1::c0",
        "paper::p2::c1",
    ]


def test_citation_grounding_with_relevant_chunk():
    ans = GeneratedAnswer(
        question="q",
        answer="Answer [paper::p1::c0]",
        cited_chunk_ids=["paper::p1::c0"],
        passages_used=["paper::p1::c0"],
    )

    assert citation_grounding(
        ans,
        ["paper::p1::c0"]
    ) == 1.0


def test_abstained_answer_has_zero_confidence():
    ans = GeneratedAnswer(
        question="q",
        answer=ABSTAIN_MARKER,
        abstained=True,
    )

    report = score_confidence(
        ans,
        [],
        score_lo=0.0,
        score_hi=10.0,
    )

    assert report.score == 0.0


def test_retrieval_gate_abstains_below_threshold():
    passages = [passage(score=2.0)]

    assert should_abstain_on_retrieval(
        passages,
        threshold=3.0
    ) is True


def test_retrieval_gate_answers_above_threshold():
    passages = [passage(score=4.0)]

    assert should_abstain_on_retrieval(
        passages,
        threshold=3.0
    ) is False


def test_abstention_counts():
    counts = AbstentionCounts()

    answered = GeneratedAnswer(
        question="q",
        answer="answer",
        abstained=False,
    )

    abstained = GeneratedAnswer(
        question="q",
        answer=ABSTAIN_MARKER,
        abstained=True,
    )

    tally_abstention(answered, True, counts)
    tally_abstention(abstained, False, counts)

    assert counts.answered_correctly == 1
    assert counts.correctly_abstained == 1


def test_threshold_requires_two_classes():
    with pytest.raises(ValueError):
        suggest_threshold([1.0, 2.0], [])



def test_citation_grounding_is_none_for_abstained_answer():
    from src.evaluation.generation_metrics import (
        citation_grounding,
    )
    from src.generation.generator import (
        GeneratedAnswer,
        ABSTAIN_MARKER,
    )

    answer = GeneratedAnswer(
        question="Question",
        answer=ABSTAIN_MARKER,
        cited_chunk_ids=[],
        raw_cited_chunk_ids=[],
        abstained=True,
        passages_used=[],
    )

    result = citation_grounding(
        answer,
        ["paper::p1::c0"],
    )

    assert result is None



def test_generator_uses_deterministic_temperature():
    from unittest.mock import Mock
    from src.generation.generator import generate_answer
    from src.retrieval.base import RetrievalResult

    client = Mock()

    block = Mock()
    block.type = "text"
    block.text = "Supported claim [paper::p1::c0]."

    response = Mock()
    response.content = [block]

    client.messages.create.return_value = response

    passages = [
        RetrievalResult(
            chunk_id="paper::p1::c0",
            paper_id="paper",
            text="Supported claim.",
            title="Paper",
            page_number=1,
            score=1.0,
            rank=1,
        )
    ]

    generate_answer(
        "Question?",
        passages,
        client=client,
    )

    kwargs = client.messages.create.call_args.kwargs

    assert kwargs["temperature"] == 0
