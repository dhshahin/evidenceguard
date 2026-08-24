"""Run the generation-phase evaluation for EvidenceGuard.

Pipeline per benchmark question:

    retrieval
    -> leakage-free abstention calibration
    -> retrieval abstention gate
    -> citation-grounded generation
    -> citation evaluation
    -> claim-level NLI verification
    -> verification gate
    -> confidence scoring

Detailed per-question results are written to JSONL for later auditing.
"""

from __future__ import annotations

import json

import torch
from anthropic import Anthropic
from sentence_transformers import CrossEncoder

from src.config import (
    PROCESSED,
    EVAL,
    RESULTS,
    TOP_K,
    NLI_MODEL,
)
from src.retrieval.base import load_chunks
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rerank_retriever import RerankRetriever
from src.evaluation.metrics import load_benchmark
from src.evaluation.calibration import leave_one_out_calibration
from src.generation.generator import (
    generate_answer,
    GeneratedAnswer,
    ABSTAIN_MARKER,
)
from src.generation.abstention import (
    top_score,
    should_abstain_on_retrieval,
)
from src.confidence.confidence import score_confidence
from src.evaluation.generation_metrics import (
    citation_validity,
    citation_grounding,
    AbstentionCounts,
    tally_abstention,
)
from src.verification.claim_verifier import (
    verify_answer,
    VerificationReport,
    build_verified_final_answer,
)
from src.evaluation.reference_alignment import (
    align_claim_to_reference,
)


VERIFICATION_TERMINOLOGY_ALIASES = {
    "LVI": "lymphovascular invasion",
    "DL": "deep learning",
}

VERIFICATION_EVIDENCE_WINDOW_SENTENCES = 3


def verify_generated_answer(
    answer,
    retrieved_results,
    nli_model,
) -> VerificationReport | None:
    """Verify an answered generation against its cited retrieved evidence."""
    if answer.abstained:
        return None

    passages = {
        result.chunk_id: result.text
        for result in retrieved_results
    }

    return verify_answer(
        answer=answer.answer,
        passages=passages,
        model=nli_model,
        entailment_threshold=0.5,
        evidence_window_sentences=VERIFICATION_EVIDENCE_WINDOW_SENTENCES,
        terminology_aliases=VERIFICATION_TERMINOLOGY_ALIASES,
    )


def build_nli_model(device):
    """Load the NLI model used for claim-level verification."""
    return CrossEncoder(
        NLI_MODEL,
        device=device,
    )


def build_retriever(chunks, device):
    """Build the retrieval and reranking stack."""
    bm25 = BM25Retriever()
    bm25.index(chunks)

    dense = DenseRetriever(
        device=device,
    )
    dense.index(chunks)

    hybrid = HybridRetriever(
        bm25,
        dense,
    )
    hybrid.index(chunks)

    return RerankRetriever(
        hybrid,
        pool=50,
        device=device,
    )


def main() -> None:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    chunks = load_chunks(
        PROCESSED / "chunks.jsonl"
    )

    benchmark_path = (
        EVAL / "benchmark_audited.jsonl"
    )

    print(
        f"Benchmark: {benchmark_path}"
    )

    benchmark = load_benchmark(
        benchmark_path
    )

    retriever = build_retriever(
        chunks,
        device,
    )

    nli_model = build_nli_model(
        device,
    )

    client = Anthropic()

    # Pass 1: retrieve evidence and construct leakage-free
    # leave-one-out calibration folds.
    all_scores = []
    answerability_labels = []
    retrieved_cache = {}

    for q in benchmark:
        res = retriever.search(
            q["question"],
            k=TOP_K,
        )

        retrieved_cache[q["qid"]] = res

        all_scores.append(
            top_score(res)
        )

        answerability_labels.append(
            bool(
                q["relevant_chunk_ids"]
            )
        )

    calibration_folds = (
        leave_one_out_calibration(
            all_scores,
            answerability_labels,
        )
    )

    print("Calibration: leave-one-out")
    print(
        "Each question is evaluated using calibration values "
        "computed from the remaining questions.\n"
    )

    counts = AbstentionCounts()

    validities = []
    groundings = []
    answered_confidences = []
    verification_support_rates = []
    reference_alignment_rates = []
    evaluation_records = []

    print(
        f"{'qid':<6}"
        f"{'abst':>6}"
        f"{'cites':>7}"
        f"{'valid':>7}"
        f"{'ground':>7}"
        f"{'verify':>8}"
        f"{'conf':>7}"
    )

    print("-" * 48)

    for i, q in enumerate(benchmark):
        res = retrieved_cache[
            q["qid"]
        ]

        is_answerable = bool(
            q["relevant_chunk_ids"]
        )

        (
            threshold,
            score_lo,
            score_hi,
        ) = calibration_folds[i]

        if should_abstain_on_retrieval(
            res,
            threshold,
        ):
            ans = GeneratedAnswer(
                question=q["question"],
                answer=ABSTAIN_MARKER,
                abstained=True,
                passages_used=[
                    r.chunk_id
                    for r in res
                ],
            )
        else:
            ans = generate_answer(
                q["question"],
                res,
                client=client,
            )

        tally_abstention(
            ans,
            is_answerable,
            counts,
        )

        conf = score_confidence(
            ans,
            res,
            score_lo,
            score_hi,
        )

        if not ans.abstained:
            answered_confidences.append(
                conf.score
            )

        validity = citation_validity(
            ans
        )

        grounding = citation_grounding(
            ans,
            q["relevant_chunk_ids"],
        )

        if validity is not None:
            validities.append(
                validity
            )

        if grounding is not None:
            groundings.append(
                grounding
            )

        verification = (
            verify_generated_answer(
                ans,
                res,
                nli_model,
            )
        )

        if verification is not None:
            verification_support_rates.append(
                verification.support_rate
            )

            verify_display = (
                f"{verification.support_rate:.2f}"
            )

            (
                verified_answer,
                post_verification_abstained,
            ) = build_verified_final_answer(
                verification
            )

            verified_claims = [
                {
                    "claim": item.claim,
                    "label": item.label,
                    "entailment_score": (
                        item.entailment_score
                    ),
                    "supported": (
                        item.supported
                    ),
                    "cited_chunk_ids": (
                        item.cited_chunk_ids
                    ),
                    "best_evidence_chunk_id": (
                        item.best_evidence_chunk_id
                    ),
                    "best_evidence_text": (
                        item.best_evidence_text
                    ),
                }
                for item
                in verification.claims
            ]

            # Reference alignment is an evaluation-only metric.
            # It is calculated only for claims that already passed
            # evidence verification, so it measures whether supported
            # claims also stay within the scope of the gold answer.
            alignment_details = []

            for item in verification.claims:
                if not item.supported:
                    continue

                alignment = align_claim_to_reference(
                    claim_text=item.claim,
                    reference_answer=q["answer"],
                    model=nli_model,
                )

                alignment_details.append(
                    {
                        "claim": item.claim,
                        "aligned": alignment.aligned,
                        "label": alignment.label,
                        "entailment_score": (
                            alignment.entailment_score
                        ),
                    }
                )

            if alignment_details:
                reference_alignment_rate = (
                    sum(
                        detail["aligned"]
                        for detail in alignment_details
                    )
                    / len(alignment_details)
                )

                reference_alignment_rates.append(
                    reference_alignment_rate
                )
            else:
                reference_alignment_rate = None

        else:
            verify_display = "-"
            verified_answer = ans.answer
            post_verification_abstained = (
                ans.abstained
            )
            verified_claims = []
            alignment_details = []
            reference_alignment_rate = None

        evaluation_records.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "reference_answer": q.get(
                    "answer"
                ),
                "answerable": (
                    is_answerable
                ),
                "retrieval_threshold": (
                    threshold
                ),
                "retrieval_top_score": (
                    top_score(res)
                ),
                "raw_answer": ans.answer,
                "raw_abstained": (
                    ans.abstained
                ),
                "raw_citations": (
                    ans.raw_cited_chunk_ids
                ),
                "valid_citations": (
                    ans.cited_chunk_ids
                ),
                "citation_validity": (
                    validity
                ),
                "citation_grounding": (
                    grounding
                ),
                "verification_support_rate": (
                    verification.support_rate
                    if verification is not None
                    else None
                ),
                "reference_alignment_rate": (
                    reference_alignment_rate
                ),
                "reference_alignment_claims": (
                    alignment_details
                ),
                "verified_claims": (
                    verified_claims
                ),
                "verified_answer": (
                    verified_answer
                ),
                "post_verification_abstained": (
                    post_verification_abstained
                ),
                "confidence": (
                    conf.score
                ),
                "retrieved_chunk_ids": [
                    item.chunk_id
                    for item in res
                ],
            }
        )

        print(
            f"{q['qid']:<6}"
            f"{str(ans.abstained):>6}"
            f"{len(ans.cited_chunk_ids):>7}"
            f"{(f'{validity:.2f}' if validity is not None else '-'):>7}"
            f"{(f'{grounding:.2f}' if grounding is not None else '-'):>7}"
            f"{verify_display:>8}"
            f"{conf.score:>7.2f}"
        )

    print("\n=== Summary ===")

    if validities:
        print(
            "Mean citation validity:  "
            f"{sum(validities) / len(validities):.2f}  "
            "(fraction of cited ids that were real)"
        )

    if groundings:
        print(
            "Mean citation grounding: "
            f"{sum(groundings) / len(groundings):.2f}  "
            "(answered questions citing a benchmark-relevant chunk)"
        )

    if verification_support_rates:
        print(
            "Mean verification support: "
            f"{sum(verification_support_rates) / len(verification_support_rates):.2f}  "
            "(fraction of generated claims supported by cited evidence)"
        )
    else:
        print(
            "Mean verification support: n/a"
        )

    if reference_alignment_rates:
        print(
            "Mean reference alignment: "
            f"{sum(reference_alignment_rates) / len(reference_alignment_rates):.2f}  "
            "(evidence-supported claims aligned with the benchmark answer)"
        )
    else:
        print(
            "Mean reference alignment: n/a"
        )

    if answered_confidences:
        print(
            "Mean confidence (answered): "
            f"{sum(answered_confidences) / len(answered_confidences):.2f}"
        )
    else:
        print(
            "Mean confidence (answered): n/a"
        )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS
        / "generation_eval_results.jsonl"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in evaluation_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"Detailed results written to: {output_path}"
    )

    print()
    print(
        counts.as_table()
    )


if __name__ == "__main__":
    main()
