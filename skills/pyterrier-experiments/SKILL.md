---
name: pyterrier-experiments
description: >-
  Loads IR test collections and evaluates PyTerrier pipelines: pt.get_dataset and
  ir_datasets ids, topics and qrels, pt.Experiment with ir_measures metrics
  (nDCG@10, AP, RR@10, recall), significance tests with multiple-testing
  correction, per-query results, saved runs, TREC DL / MS MARCO / Robust04 / BEIR
  conventions, whole benchmark suites with SuiteEval (BEIR, NanoBEIR, LoTTE,
  BRIGHT), and parameter tuning with GridSearch. Use when asked to evaluate,
  benchmark or compare retrieval runs, compute metrics, test significance, tune
  parameters, or load a dataset in PyTerrier, even if pt.Experiment is not named.
  Not for building indexes or choosing retrievers.
compatibility: pyterrier>=1.1 (checked with 1.1.2, ir_datasets 0.6.3, ir_measures 0.4.3, suiteeval 0.1.8)
---

# PyTerrier experiments

Assumes the `pyterrier-basics` skill.

Docs: [Datasets](https://pyterrier.readthedocs.io/en/latest/datasets.html) ·
[Experiments](https://pyterrier.readthedocs.io/en/latest/experiments.html) ·
[Tuning](https://pyterrier.readthedocs.io/en/latest/tuning.html) ·
[SuiteEval](https://pyterrier.readthedocs.io/en/latest/ext/suiteeval/index.html) ·
[ir_datasets catalogue](https://ir-datasets.com/) ·
[ir_measures measures](https://ir-measur.es/en/latest/)

## Quick start

```python run
import pyterrier as pt
from pyterrier.measures import *

dataset = pt.get_dataset("irds:vaswani")
topics = dataset.get_topics()
qrels = dataset.get_qrels()

index = pt.terrier.TerrierIndex.example()   # vaswani: fine for a task on vaswani, never for comparing with papers
results = pt.Experiment(
    {"BM25": index.bm25(), "DPH": index.dph()},
    topics,
    qrels,
    eval_metrics=[nDCG@10, AP(rel=1), RR@10, R@1000],
    baseline="BM25",
    correction="holm",
    round=4,
)
print(results)
assert {"name", "nDCG@10", "nDCG@10 p-value corrected"} <= set(results.columns)
```

## Datasets

- Look up dataset ids on [ir-datasets.com](https://ir-datasets.com/) and prefix them with `irds:`, e.g.
  `pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")`.
- Before building a pipeline, print `topics.columns`, `qrels.columns` and one document from
  `get_corpus_iter()`. Field names differ between collections.

| Method | Returns |
|---|---|
| `get_topics(variant)` | Q frame: `qid`, `query` |
| `get_qrels()` | `qid`, `docno`, `label` (some collections also have `iteration`) |
| `get_corpus_iter()` | iterator of dicts with `docno` and text fields, for indexing |
| `text_loader()` | transformer that adds document text to results, for rerankers |

- **Topics with several fields.** Collections such as Robust04 and TREC-COVID have
  `title`/`description`/`narrative` fields. Without a variant, `get_topics()` only prints a message
  and returns those fields with **no `query` column**. Pass the field you want:
  `get_topics("title")`. BEIR collections have no `title` variant; their query text is `text`.
- ir_datasets ids are mapped: `doc_id` becomes `docno` and `query_id` becomes `qid`. The text
  field keeps its own name (`text`, `body`, `title`, `abstract`, ...).
- Licensed corpora (e.g. Robust04 documents) still have public topics and qrels. You can evaluate a
  prebuilt index or a saved run without the documents.
- NanoBEIR datasets need `pyarrow` (`pip install 'ir_datasets[pyarrow]'`).

## Evaluating with pt.Experiment

```python
pt.Experiment(
    systems,          # list of transformers or result frames, or a dict {name: system}
    topics,
    qrels,
    eval_metrics=[nDCG@10, AP(rel=2)],   # required, and named eval_metrics, not metrics
    names=[...],      # only with a list; a dict supplies the names
    baseline=0,       # index into the list, or a name when systems is a dict
    correction="holm",
    perquery=False,   # True: one row per query and measure; "both": (averages, per_query)
    save_dir=None,    # folder must already exist
    save_mode="warn", # "reuse" | "overwrite" | "error" | "warn"
    plan="linear",    # "tree" shares common pipeline prefixes, but not with save_dir
)
```

| Need | Use |
|---|---|
| Measures | `from pyterrier.measures import *`, then e.g. `nDCG@10`, `AP(rel=2)`, `RR(rel=2)@10`, `R@1000`, `P@10`. Syntax on [ir-measur.es](https://ir-measur.es/en/latest/). `MAP` is accepted and is printed as `AP` |
| Significance | `baseline=` plus `correction="holm"` (or `"bonferroni"`, `"hs"`; any statsmodels `multipletests` method). The default test is a paired t-test (`test="t"`). Adds `+`, `-`, `p-value`, `reject` and `p-value corrected` columns per measure. The baseline row's significance columns are NaN (its measures print normally), and step-down corrections such as Holm can give several systems the same corrected p-value; both are expected |
| Per-query scores | `perquery=True`: columns `name`, `qid`, `measure`, `value`. `perquery="both"` returns `(averages, per_query)` |
| Mean response time | add `"mrt"` to `eval_metrics` |
| Evaluate an existing run | pass the results frame as a system, or `pt.Evaluate(run, qrels, metrics=[...])` |
| Keep runs on disk | `os.makedirs(d, exist_ok=True)` then `save_dir=d`. Writes `<name>.res.gz`, `aggregated.csv`, `perquery.csv`. Load a run back with `pt.io.read_results(path)` (`qid, docno, rank, score`; there is no `pt.load_run`) |
| Faster when systems share a prefix | `plan="tree"` (PyTerrier warns when this applies) |

- Pick measures from the task or the benchmark's conventions, not from an example. For TREC DL,
  MS MARCO, Robust04 and BEIR conventions, read [references/benchmarks.md](references/benchmarks.md).
- Use `rel=` whenever labels are graded and the measure is binary and the benchmark sets a threshold. For
  example, TREC DL `AP` and `RR` count `label >= 2` as relevant, so write `AP(rel=2)` and `RR(rel=2)@10`.
  BEIR counts `label >= 1`, the default, so plain `R@100` is right there.
- Compare every system in **one** `pt.Experiment` call, so they are evaluated on the same topics and
  tested against the same baseline.
- With `save_mode="reuse"`, a saved run is found **by system name**. Rename a system when you change
  its pipeline, or delete its `.res.gz`.

```python run
per_query = pt.Experiment(
    [index.bm25()], topics, qrels, eval_metrics=[nDCG@10], names=["BM25"], perquery=True
)
assert list(per_query.columns) == ["name", "qid", "measure", "value"]
```

## Whole benchmark suites with SuiteEval

SuiteEval runs one pipeline factory over every dataset in a suite. It builds one index per corpus and
returns one table with a `dataset` column and an `Overall` row.

```python
import os
import pyterrier as pt
from pyterrier.measures import *
from suiteeval import NanoBEIR

def pipelines(context):
    # Called once per corpus. context.path is a str
    index = pt.terrier.TerrierIndex(os.path.join(context.path, "index.terrier"))
    if not index.built():                          # needed when index_dir is reused
        # explicit meta: index.index() guesses lengths from the first document and truncates
        index.indexer(meta={"docno": 200}).index(context.get_corpus_iter())
    yield index.bm25(), "BM25"                      # always name pipelines
    yield index.dph(), "DPH"

results = NanoBEIR(
    pipelines,
    eval_metrics=[nDCG@10],     # pin the measures
    index_dir="indices",        # reusable indexes, one folder per corpus
    save_dir="runs",            # reusable runs, per dataset and pipeline name
    # subset="nano-beir/scifact",  # one dataset only, while developing
)
```

| Suite | Datasets | Default measure |
|---|---|---|
| `BEIR` | 14 BEIR datasets plus CQADupStack (no SciDocs) | nDCG@10 |
| `NanoBEIR` | 13 small BEIR samples | nDCG@10 |
| `Lotte` | 6 domains × forum/search | Success@10 |
| `BRIGHT` | 12 reasoning-intensive datasets | nDCG@10 |
| `MSMARCOPassage` / `MSMARCODocument` | TREC DL 2019, 2020, 2022 | nDCG@10 |

- Develop with `subset="..."` and NanoBEIR. A full `BEIR` run indexes about 15 corpora, and the
  MS MARCO suites include MS MARCO v2 (about 138M passages).
- Without `index_dir`, indexes go into temporary folders that are **never deleted**.
- The `Overall` row is a **geometric mean** across datasets. BEIR's 12 CQADupStack forums are merged by
  geometric mean too. Neither matches arithmetic means in papers, so say which one you report.
- Other keyword arguments, such as `baseline=` and `correction=`, are passed through to `pt.Experiment`.
  `baseline=` must be an **int** (the position in the factory's yield order, e.g. `baseline=0`): SuiteEval
  passes the pipelines as a list, so a name raises `TypeError: baseline should be an int`.
- With `save_dir`, pipeline names become file names and may not contain `:"%/<>^|?`. Name a reranked
  system `BM25+monoELECTRA`, not `BM25>>monoELECTRA`.
- With `save_dir`, a run whose `.res.gz` already exists is replayed from disk: the pipeline does not
  execute (scorer caches and `Lazy` models print no hit/miss output) and the replay ignores your
  `pt.Experiment` kwargs, so `baseline=` significance can be inconsistent. Pass `save_mode="overwrite"`
  to execute the pipelines: to exercise a cache, or for per-query results and significance tests
  (references/suiteeval.md, "Known problems").

For the full dataset list, the caching layout, custom suites and known bugs, read
[references/suiteeval.md](references/suiteeval.md).

## Tuning parameters

```python
bm25 = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b": 0.75})  # initial value required
tuned = pt.GridSearch(
    bm25,
    {bm25: {"bm25.b": [0.3, 0.5, 0.75, 0.9]}},
    train_topics,
    qrels[qrels.qid.isin(train_topics.qid)],   # only the tuning topics' qrels (see below)
    metric="map",          # trec_eval-style string, e.g. "map", "ndcg_cut_10"
)
```

- `metric=` must be a trec_eval-style string. Measure objects such as `nDCG@10` raise `KeyError`,
  and the string `"nDCG@10"` raises `ValueError`.
- Every tuned parameter needs an initial value in `controls=` when the retriever is created.
- To report the best value, pass `return_type="best_setting"`: it returns `(score, [(transformer, param,
  value), ...])`; read the values with `{p: v for _, p, v in settings}`. The default returns the pipeline set
  to the best values, not the values themselves (tuning.md §1).
- Pass only the tuning topics' qrels. `GridSearch` and `GridScan` average over every qid in `qrels`
  (unlike `pt.Experiment`, they don't filter by topics), so full qrels with a topic subset scale the
  score by len(train_topics) / len(qrels qids). The best setting is unchanged; the printed score is wrong.
- Tune on training or validation topics, then evaluate the tuned pipeline once on test topics. Never
  tune on the topics you report. With no train/test split (vaswani, Robust04), tune on one half of the
  topics and report on the other, or use `KFoldGridSearch` (tuning.md §4).
- For full signatures, `GridScan`, `KFoldGridSearch` and known problems, read
  [references/tuning.md](references/tuning.md).

## Checklist for an evaluation

```
- [ ] Dataset id checked on ir-datasets.com, and the topic variant chosen.
- [ ] topics has `qid` and `query`; qrels has `qid`, `docno` and `label`, all ids as strings.
- [ ] Measures follow the benchmark's conventions, including `rel=` for graded labels.
- [ ] All systems in one pt.Experiment, with a baseline and a correction.
- [ ] Report the dataset id, topic variant, measures, cut-offs and the correction used.
```

## Don't

- Don't pass `metrics=` to `pt.Experiment`. The argument is `eval_metrics`.
- Don't pass `correction=` without `baseline=`.
- Don't use `get_topics()` without a variant on collections with several query fields.
- Don't use binary measures on graded qrels without `rel=`.
- Don't combine `save_dir=` with `plan="tree"`.
- Don't reuse a system name for a changed pipeline when `save_mode="reuse"`.
- Don't compare numbers from different topic sets, or from vaswani, with published results.
- Don't tune on the test topics.
- Don't pass a measure object to `pt.GridSearch(metric=...)`. Use a string such as `"map"`.
- Don't give `pt.GridSearch` or `pt.GridScan` the full qrels with a subset of topics. Filter the qrels to those topics.
- Don't call SuiteEval suites as classes (`BEIR()`). They are ready-made instances: `BEIR(pipelines)`.
- Don't build an index unconditionally inside a SuiteEval factory when `index_dir` is set. Check
  `index.built()` first.

## Reference files

| Read | When |
|---|---|
| [references/benchmarks.md](references/benchmarks.md) | evaluating on TREC DL, MS MARCO, Robust04 or BEIR, or checking a baseline against published numbers |
| [references/suiteeval.md](references/suiteeval.md) | running a whole suite, defining a custom suite, or debugging SuiteEval caching |
| [references/tuning.md](references/tuning.md) | tuning parameters with GridSearch, GridScan or KFoldGridSearch |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: Experiment() missing 1 required positional argument: 'eval_metrics'` | measures passed as `metrics=` | `eval_metrics=[...]` |
| `AssertionError: invalid param cutoff=<object object at …>` from inside `ir_measures` | a measure with a mandatory cutoff used bare: `Recall`, `R` or `P` | write the cutoff: `R@1000`, `P@10`; `nDCG`, `AP` and `RR` are fine bare |
| `ValueError: Requested multiple testing correction, but no baseline was specified.` | `correction=` without `baseline=` | add `baseline=` |
| `ValueError: save_dir runs does not exist` | `save_dir` folder missing | `os.makedirs(save_dir, exist_ok=True)` first |
| `ValueError: save_dir is not yet supported for tree execution plan` | `save_dir` with `plan="tree"` | drop one of them |
| Message `There are multiple query fields available`, then `KeyError: 'query'` | multi-field topics loaded without a variant | `get_topics("title")` |
| Unchanged scores after changing a pipeline | `save_mode="reuse"` loaded the old run by name | rename the system or delete its `.res.gz` |
| No cache hit/miss output on a SuiteEval rerun, or inconsistent significance columns | the saved `.res.gz` was replayed without executing the pipeline | `save_mode="overwrite"` |
| `UserWarning: There are shared pipeline components among N pipelines` | systems share a prefix | `plan="tree"` (without `save_dir`) |
| `ImportError: This dataset requires pyarrow` | NanoBEIR and some other datasets store data as parquet | `pip install 'ir_datasets[pyarrow]'` |
| Disk fills up after SuiteEval runs | temporary indexes are never deleted | pass `index_dir=` and delete it yourself |
| `KeyError: nDCG@10` (or `AP`) from `GridSearch` | measure object passed as `metric=` | `metric="ndcg_cut_10"` or `metric="map"` |
| `ValueError: Invalid parameter name bm25.b for transformer` | parameter has no initial value | `pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b": 0.75})` |
| GridSearch's "Best …" score is far below `pt.Experiment` on the same topics | full qrels passed with a topic subset; GridSearch averages over every qrels qid | `qrels[qrels.qid.isin(topics.qid)]` |
| `TypeError: baseline should be an int when retr_systems is a list` from a suite | `baseline="BM25"` passed to SuiteEval | `baseline=0`, the index in the factory's yield order |
| `ValueError: Name contains bad characters and save_dir is set` | a pipeline name with `>`, `:`, `/` or similar | use letters, digits and `.-_+` |
| Scores near 0 for every system | `qid`/`docno` types or values don't match the qrels | check ids are strings and `docno` values match |
| The results table prints with `...` and `[N rows x M columns]` | pandas truncates the significance columns | `pd.set_option("display.max_columns", None, "display.width", 250)` before printing |
