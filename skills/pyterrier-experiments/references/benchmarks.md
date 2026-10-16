# Benchmark conventions

Dataset ids, topic counts and label scales below were checked with ir_datasets 0.6.3. Reference scores
are **indicative**: always confirm them against the source given before using one as a gate or in a
report.

## Contents
1. MS MARCO passage and TREC Deep Learning
2. Robust04
3. BEIR
4. Using reference scores as a gate

## 1. MS MARCO passage and TREC Deep Learning

| Topic set | Dataset id | Topics | Labels | Usual measures |
|---|---|---|---|---|
| MS MARCO dev (small) | `irds:msmarco-passage/dev/small` | 6,980 | binary (1) | `RR@10`; also `R@1000` |
| TREC DL 2019 passage | `irds:msmarco-passage/trec-dl-2019/judged` | 43 | graded 0–3 | `nDCG@10`, `AP(rel=2)`, `RR(rel=2)@10`, `R(rel=2)@1000` |
| TREC DL 2020 passage | `irds:msmarco-passage/trec-dl-2020/judged` | 54 | graded 0–3 | as DL 2019 |
| TREC DL 2019 / 2020 document | `irds:msmarco-document/trec-dl-2019/judged`, `.../trec-dl-2020/judged` | | graded | `nDCG@10`, `AP(rel=2)` |

- Use the `/judged` variants. They contain only queries that have judgements. The unfiltered sets
  contain hundreds of unjudged queries that would count as zero.
- nDCG@10 uses the graded labels as they are.
- AP, RR and recall binarise the labels: TREC DL counts **label ≥ 2** as relevant for these, so write
  `rel=2`. Without it, a label of 1 counts as relevant and scores are higher than published ones.
- MAP is computed over the full ranking, so it depends on how many results the first stage returns.
  Report the depth (usually 1000).
- The corpus is shared: one index of `irds:msmarco-passage` serves dev, DL 2019 and DL 2020.
- The dev set is large. Evaluate a reranker on a sample only while developing, and say so.

## 2. Robust04

| Item | Value |
|---|---|
| Dataset id | `irds:disks45/nocr/trec-robust-2004` |
| Topics | `get_topics("title")` (or `"description"`). There is no `query` column without a variant |
| Labels | graded 0–2 |
| Documents | licensed (TREC disks 4 and 5). Topics and qrels download without a licence |
| Usual measures | `AP` (MAP), `P@20`, `nDCG@20` |

- There is no standard train/test split. Neural work commonly uses five folds, so use
  `pt.KFoldGridSearch` or the folds of the paper you compare with, and report which.
- With graded labels, binary measures count any label ≥ 1 as relevant by default. Confirm the
  threshold the comparison paper used.

## 3. BEIR

- Dataset ids are `irds:beir/<name>` or `irds:beir/<name>/test`. Check the exact id on
  [ir-datasets.com/beir](https://ir-datasets.com/beir.html).
- The standard measure is nDCG@10 per dataset. Report each dataset, and say how you average them
  (arithmetic mean is the BEIR convention).
- The published BEIR BM25 baseline is Anserini/Pyserini BM25: k1=0.9, b=0.4, Porter stemming, Lucene
  English stopwords (values in the `pyterrier-retrieval` skill, references/defaults.md). Terrier's
  built-in BM25 uses k1=1.2, b=0.75, so for BEIR-family comparisons (BEIR, NanoBEIR) pass
  `index.bm25(k1=0.9, b=0.4)` explicitly.
- Recall is plain `R@100` or `R@1000`: `ir_measures` counts labels >= 1 as relevant, which is the BEIR
  convention. Don't add `rel=2`.
- CQADupStack is 12 sub-forums. The convention is to average them into a single CQADupStack score.
- SciDocs, BioASQ, Signal-1M, TREC-NEWS and Robust04 are sometimes left out, because some need
  licences. State exactly which datasets your average covers.
- Topics use the `text` field. Some BEIR datasets have extra query fields, so check `topics.columns`.

## 4. Using reference scores as a gate

Before adding stages to a pipeline, check that the first stage scores close to a known reference. A
large gap usually means a wrong dataset id, a missing `rel=`, a stemming or tokenisation mismatch, or
different BM25 parameters.

Reference scores found on prebuilt index cards on Hugging Face (BM25, nDCG@10):

| Collection | Index | nDCG@10 | Source |
|---|---|---|---|
| TREC DL 2019 passage | `macavaney/msmarco-passage.pisa` | 0.4989 | the index's Hugging Face card |
| TREC DL 2019 passage | `macavaney/msmarco-passage.anserini` | 0.5121 | the index's Hugging Face card |
| BEIR FiQA (test) | `pyterrier/fiqa.terrier` | 0.2526 | the index's Hugging Face card |
| BEIR FiQA (test) | `pyterrier/fiqa.pisa` | 0.2411 | the index's Hugging Face card |

- These are a starting point. Read the card itself before relying on a number, because cards can
  change.
- For other collections and baselines, use the Pyserini reproduction matrix
  (https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html and the BEIR page linked from it),
  and note that Anserini's BM25 parameters differ from Terrier's and PISA's.
- Systems that are implemented differently can differ by about ±0.02 nDCG@10. Investigate a larger gap
  before going on.
