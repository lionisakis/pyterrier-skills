---
name: reproduce-ir-paper
description: Reproduce a neural information-retrieval paper's reported results — build a first-stage retrieval run, rerank it with a cross-encoder, evaluate with the paper's metrics, and test significance. Use when asked to reproduce, replicate or verify results to evaluate a reranker against published numbers.
---

# Reproducing a neural IR paper

Tooling and conventions for turning a paper's reported table into a run you can compare against. This
covers *how to do it correctly*; the paper you are reproducing supplies *what* to do — its method, its
normalisation, its cutoffs. Read the paper for those; do not guess.

## 1. ⚠️ Read this before writing any code

**Never run on a toy collection and compare the result to a paper's table.**

`vaswani`, `TerrierIndex.example()`, and anything else named *sample*, *example* or *demo* exist to check
that your tooling executes. They are a few thousand documents from a different domain. Numbers from them
have no relationship to a paper's numbers, and comparing the two produces a meaningless
verdict.

Any code you find in a skill, tutorial or README — including this one — is illustrating **a technique**.
The collection, the topic set, the rerank depth and the metrics come from **the paper you are
reproducing**, never from the example. Copying an example's *setup* instead of its *method* is the single
most common way a reproduction ends up plausible and wrong.

## 2. First pass candidate retrieval

Check carefully what the canididate idenfication retriever is from the original paper.
Sparse: use Terrier (`pt.terrier.TerrierIndex`) or Pisa retrievers (pyterrier_pisa);
Dense: use pyterrier_dr.

**BM25 is not one algorithm.** Terrier's defaults (k1=1.2, b=0.75) differ from Anserini/Lucene's tuned
MS MARCO settings (k1=0.82, b=0.68), and tokenisation and stopword lists differ too. Scores are on
different scales. This matters enormously if a BM25 score is being used as a *model input feature*
rather than only for ordering — a distribution shift there corrupts the input rather than merely
reordering candidates. Match the paper's retriever, or state the deviation explicitly.

You should ensure that the effectiveness of your baseline first pass candidate retrieval is close or matches the original paper, e.g. in terms of NDCG and MAP.

Use prebuilt indices, e.g. downloaded as artifacts from HuggingFace (c.f. `pt.Artifact.from_hf(url)`) rather
than building a new index for a corpus.

## 3. Reranking

If you are developing a reranker, see the skills named reranking-experiments-skills.

## 4. Evaluating

Use `pt.Experiment()` to evaluate the pipeline on the chose dataset, using the appropriate topics and qrels. pt.Experiment is described in the skills named reranking-experiments-skills. Refer to any skills that are appropriate to the chosen dataset.

## 4. Reporting

- If you have been given access to published numbers, keep **published** and **reproduced** figures in separate columns and never overwrite one with the other. Show the delta.
- Distinguish **matching the numbers** from **reproducing the claim**. A run can land within 0.01 of every
  published figure and still fail to reproduce a paired effect — and the paired effect is usually what
  the paper actually asserts.
- **Report faithfully.** If the numbers don't match, report them as they are and give your best
  explanation. Never adjust an evaluation, drop queries, or switch metrics to move results toward the
  published values. A negative result, clearly reported with its likely cause, is a valid finding; a
  massaged positive one is worthless.
- Record **assumptions** wherever the paper underspecifies (epochs, negative sampling, exact
  retriever settings are commonly missing) and note any **discrepancy between the paper and its released
  artifacts** — these diverge more often than you would expect.

