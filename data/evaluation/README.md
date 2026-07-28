# Benchmark: how to build it

The benchmark is what turns eyeballing into measurement. It is the most
valuable artifact in the whole project, because every metric (Recall@k,
MRR, nDCG, later citation accuracy) is computed against it.

## Format

`data/evaluation/benchmark.jsonl` — one JSON object per question:

```json
{
  "qid": "q001",
  "question": "Which clinical features were selected as key predictors of breast cancer recurrence?",
  "answer": "Regional lymph node positivity, ER status, Ki-67, lymphovascular invasion, tumor size, and age at diagnosis.",
  "relevant_chunk_ids": ["cmar-17-917::p9::c0", "cmar-17-917::p1::c2"],
  "supporting_papers": ["cmar-17-917"],
  "notes": "LASSO-selected 6 predictors; stated in Key Predictors section."
}
```

Field meaning:
- **qid** — stable id (q001, q002, ...).
- **question** — a real research question answerable from your corpus.
- **answer** — the correct answer, in your own words, extracted from the
  papers. This is your reference/gold answer for later answer-quality
  metrics. Keep it short and factual.
- **relevant_chunk_ids** — THE KEY FIELD. The chunk_ids that actually
  contain the evidence. This is what Recall@k / MRR / nDCG are scored
  against. Find these by searching your own corpus (see workflow below).
- **supporting_papers** — which paper(s) the answer comes from. Lets you
  sanity-check and analyze per-paper.
- **notes** — anything for your own error analysis later.

## Question types to include (aim for a mix)

1. **Single-fact** — answer sits in one place. ("What AUC did the ensemble
   model achieve?")
2. **List/aggregation** — answer spans several chunks. ("Which features were
   selected?")
3. **Cross-paper / comparison** — needs evidence from 2+ papers. ("Do the
   studies agree on whether lymph node status predicts recurrence?")
4. **Insufficient-evidence** — deliberately ask something the corpus does
   NOT answer, with `relevant_chunk_ids: []`. These test whether the
   system correctly abstains instead of hallucinating. Include ~15-20% of
   these; they are essential for the abstention metric.

## Workflow for finding relevant_chunk_ids (semi-automated)

You do NOT annotate by hand from scratch. Use the retriever as an
assistant, then verify:

1. Write the question.
2. Run it through hybrid retrieval with a large k (say 15).
3. Read the returned chunks. Mark the ones that truly contain the answer.
4. Record their chunk_ids in `relevant_chunk_ids`.
5. Write the reference answer from those chunks.

This is far faster than reading all 265 chunks, and it is legitimate: you
are the judge, the retriever just proposes candidates. Note the one bias
to watch — if you only ever label chunks your retriever surfaces, you may
miss relevant chunks it never retrieves. Mitigate by occasionally keyword-
searching the raw text for a question's key terms too.

## How many

- Minimum useful: 30 questions.
- Target for the report: 50-100 (the plan's number).
- Start with 10 to get the evaluation harness running, then grow.

## Single-annotator honesty

You are one annotator, so there is no inter-annotator agreement. Be
transparent about this in the report. One cheap mitigation: re-label a
random 10 questions a week apart and report your own consistency
(intra-annotator agreement). Reviewers respect the acknowledged limitation
far more than a hidden one.
