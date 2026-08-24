"""Citation-grounded answer generation.

Given a question and a set of retrieved passages, produce an answer that:
  1. uses ONLY the supplied passages (no outside knowledge),
  2. cites the supporting chunk_id(s) inline for every claim,
  3. explicitly abstains if the passages do not contain the answer.

The generator never sees the corpus — only the passages retrieval selected.
This is what makes every answer traceable to evidence and is the core of
the "explainable / grounded" guarantee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from anthropic import Anthropic

from src.config import GEN_MODEL, GEN_MAX_TOKENS
from src.retrieval.base import RetrievalResult

# Sentinel the model is told to emit when evidence is insufficient.
ABSTAIN_MARKER = "INSUFFICIENT_EVIDENCE"


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    raw_cited_chunk_ids: list[str] = field(default_factory=list)
    abstained: bool = False
    passages_used: list[str] = field(default_factory=list)  # chunk_ids offered to the model
    raw_model_output: str = ""


def _format_passages(passages: list[RetrievalResult]) -> str:
    """Render passages with their chunk_id labels for the prompt."""
    blocks = []
    for p in passages:
        blocks.append(f"[{p.chunk_id}] (from \"{p.title}\", p.{p.page_number})\n{p.text}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "You are a careful scientific evidence assistant. You answer questions "
    "using ONLY the provided passages. You never use outside knowledge. "
    "Use only facts that are directly supported by the provided evidence. "
    "Do not infer rankings from list order or presentation order unless the "
    "evidence explicitly states the ranking. "
    "Do not expand abbreviations unless the provided evidence explicitly defines "
    "the abbreviation. Preserve the terminology used in the source whenever possible. "
    "Use one factual claim per sentence. Do not combine multiple independently "
    "verifiable facts into a single sentence. "
    "Every factual claim must be followed immediately by the chunk_id(s) "
    "that support it, in square brackets, e.g. [paper::p3::c1]. "
    "When the answer contains a list, write each list item as a separate sentence "
    "and place its supporting chunk_id(s) immediately after that sentence. "
    "Do not use one citation at the end of a paragraph to support multiple claims. "
    "Answer the question directly and concisely. Do not add extra findings that are "
    "not needed to answer the question. "
    f"If the passages do not contain enough information to answer, reply with "
    f"exactly '{ABSTAIN_MARKER}' and nothing else. Do not guess."
)


USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Answer the question using only these passages. Use one factual claim per sentence. "
    "Cite the supporting chunk_id(s) immediately after each claim. "
    "If the answer contains multiple items or findings, express them as separate "
    "atomic claims rather than combining them into one sentence. "
    "If the passages are insufficient, reply with exactly "
    f"'{ABSTAIN_MARKER}'."
)


def _extract_all_cited_ids(text: str) -> list[str]:
    """Extract every citation-like id emitted by the model.

    Citations are retained before validation so invalid or fabricated
    references remain available for evaluation.
    """
    found = re.findall(r"\[([^\]]+)\]", text)
    ids: list[str] = []

    for raw in found:
        for piece in raw.split(","):
            cid = piece.strip()
            if cid and cid not in ids:
                ids.append(cid)

    return ids


def _extract_cited_ids(text: str, valid_ids: set[str]) -> list[str]:
    """Return citations that correspond to passages actually offered."""
    return [
        cid
        for cid in _extract_all_cited_ids(text)
        if cid in valid_ids
    ]


def generate_answer(
    question: str,
    passages: list[RetrievalResult],
    client: Anthropic | None = None,
) -> GeneratedAnswer:
    """Generate a citation-grounded answer (or abstain) from passages."""
    offered = [p.chunk_id for p in passages]

    # No passages at all -> abstain without calling the model.
    if not passages:
        return GeneratedAnswer(
            question=question, answer=ABSTAIN_MARKER, abstained=True,
            passages_used=offered, raw_model_output="")

    client = client or Anthropic()
    msg = client.messages.create(
        model=GEN_MODEL,
        max_tokens=GEN_MAX_TOKENS,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(
                question=question, passages=_format_passages(passages)),
        }],
    )
    out = "".join(block.text for block in msg.content if block.type == "text").strip()

    abstained = out.strip() == ABSTAIN_MARKER

    if abstained:
        raw_cited = []
        cited = []
    else:
        raw_cited = _extract_all_cited_ids(out)
        cited = [cid for cid in raw_cited if cid in set(offered)]

    return GeneratedAnswer(
        question=question,
        answer=out,
        cited_chunk_ids=cited,
        raw_cited_chunk_ids=raw_cited,
        abstained=abstained,
        passages_used=offered,
        raw_model_output=out,
    )
