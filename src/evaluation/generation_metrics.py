"""Evaluation for the generation phase.

Retrieval metrics (recall, MRR) measure whether the right passages were
found. These metrics measure whether the *answer* behaves trustworthily:

  * citation_validity   — of the ids the answer cites, what fraction are
                          real passages that were actually offered?
                          (fabricated citations are the key failure mode.)
  * citation_grounding  — did the answer cite at least one of the passages
                          the benchmark marked relevant? (a weak proxy for
                          "grounded in the right evidence".)
  * abstention metrics  — on questions the benchmark marks unanswerable
                          (no relevant chunks), did the system correctly
                          abstain? and did it correctly ANSWER the
                          answerable ones? Reported as a small confusion
                          matrix so over- and under-abstention are visible.

Everything here is computed against the existing benchmark.jsonl, so no new
labels are needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.generation.generator import GeneratedAnswer


def citation_validity(ans: GeneratedAnswer) -> float | None:
    """Fraction of model-emitted citation ids that were actually offered.

    Raw citations are used so invalid references remain visible during
    evaluation rather than being removed before scoring.
    """
    if ans.abstained:
        return None

    citations = (
        ans.raw_cited_chunk_ids
        if ans.raw_cited_chunk_ids
        else ans.cited_chunk_ids
    )

    if not citations:
        return None

    offered = set(ans.passages_used)
    valid = sum(1 for c in citations if c in offered)

    return valid / len(citations)


def citation_grounding(ans: GeneratedAnswer, relevant_ids: list[str]) -> float | None:
    """1.0 if the answer cited any benchmark-relevant chunk, else 0.0.

    None for abstention questions (no relevant set to check against).
    """
    if not relevant_ids:
        return None
    if ans.abstained:
        return None
    return 1.0 if any(c in relevant_ids for c in ans.cited_chunk_ids) else 0.0


@dataclass
class AbstentionCounts:
    # answerable questions
    answered_correctly: int = 0    # answerable & answered
    wrongly_abstained: int = 0     # answerable but abstained (over-caution)
    # unanswerable questions
    correctly_abstained: int = 0   # unanswerable & abstained
    wrongly_answered: int = 0      # unanswerable but answered (hallucination risk)

    @property
    def abstention_precision(self) -> float | None:
        denom = self.correctly_abstained + self.wrongly_abstained
        return self.correctly_abstained / denom if denom else None

    @property
    def abstention_recall(self) -> float | None:
        denom = self.correctly_abstained + self.wrongly_answered
        return self.correctly_abstained / denom if denom else None

    def as_table(self) -> str:
        lines = [
            "Abstention confusion matrix:",
            f"  answerable   -> answered:  {self.answered_correctly}",
            f"  answerable   -> abstained: {self.wrongly_abstained}  (over-caution)",
            f"  unanswerable -> abstained: {self.correctly_abstained}",
            f"  unanswerable -> answered:  {self.wrongly_answered}  (hallucination risk)",
        ]
        p, r = self.abstention_precision, self.abstention_recall
        lines.append(f"  abstention precision: {p:.2f}" if p is not None else "  abstention precision: n/a")
        lines.append(f"  abstention recall:    {r:.2f}" if r is not None else "  abstention recall:    n/a")
        return "\n".join(lines)


def tally_abstention(ans: GeneratedAnswer, is_answerable: bool, counts: AbstentionCounts) -> None:
    if is_answerable:
        if ans.abstained:
            counts.wrongly_abstained += 1
        else:
            counts.answered_correctly += 1
    else:
        if ans.abstained:
            counts.correctly_abstained += 1
        else:
            counts.wrongly_answered += 1
