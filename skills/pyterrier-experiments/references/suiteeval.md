# SuiteEval reference

Checked with suiteeval 0.1.8 (PyPI), PyTerrier 1.1.2 and ir_datasets 0.6.3.
Source: https://github.com/Parry-Parry/suiteeval · Paper: "SuiteEval: Simplifying Retrieval Benchmarks"
(ECIR 2026 demo, arXiv 2602.18107).

## Contents
1. Install
2. How a suite runs
3. Suites, datasets and measures
4. Caching
5. Arguments
6. Custom suites
7. Known problems

## 1. Install

```bash
pip install suiteeval
pip install 'ir_datasets[pyarrow]'   # NanoBEIR and other parquet-based datasets
```

- Requires Python 3.10 or newer.
- BRIGHT needs ir_datasets 0.6 or newer.
- Install whatever your pipelines use (pyterrier-pisa, pyterrier-dr, pyterrier-t5, ...) separately.

## 2. How a suite runs

1. The suite groups its datasets by corpus, so datasets that share documents share one index.
2. For each corpus it creates a `DatasetContext` and calls your factory **once**:
   - `context.path` is a **str** folder for the index. Build paths with `os.path.join`, not `/`.
   - `context.get_corpus_iter()` gives the documents for indexing.
   - `context.text_loader()` gives a transformer that adds document text, for rerankers.
   - `context.dataset` is the corpus-level PyTerrier dataset.
3. Each pipeline the factory yields is run on every dataset of that corpus with `pt.Experiment`.
4. The results are concatenated with a `dataset` column, and `Overall` rows are appended.

The factory may yield or return any of these:
- a transformer;
- a `(transformer, "name")` pair;
- a list of transformers;
- a `(list, [names])` pair.

Name every pipeline: unnamed pipelines are never cached.

The suites are ready-made instances, not classes: call `BEIR(pipelines)`, not `BEIR()(pipelines)`.
Import them with `from suiteeval import BEIR, NanoBEIR, Lotte, BRIGHT, MSMARCOPassage, MSMARCODocument`.

## 3. Suites, datasets and measures

| Suite | ir_datasets ids | Measure |
|---|---|---|
| `BEIR` | `beir/arguana`, `beir/climate-fever`, `beir/cqadupstack/{android,english,gaming,gis,mathematica,physics,programmers,stats,tex,unix,webmasters,wordpress}`, `beir/dbpedia-entity/test`, `beir/fever/test`, `beir/fiqa/test`, `beir/hotpotqa/test`, `beir/msmarco/test`, `beir/nfcorpus/test`, `beir/nq`, `beir/quora/test`, `beir/scifact/test`, `beir/trec-covid`, `beir/webis-touche2020/v2`. **No `beir/scidocs`.** | nDCG@10 |
| `NanoBEIR` | `nano-beir/{arguana, climate-fever, dbpedia-entity, fever, fiqa, hotpotqa, msmarco, nfcorpus, nq, quora, scidocs, scifact, webis-touche2020}` | nDCG@10 |
| `Lotte` | `lotte/{lifestyle,pooled,recreation,science,technology,writing}/test/{forum,search}` | Success@10, plus the dataset's official measures (Success@5) |
| `BRIGHT` | `bright/{aops, biology, earth-science, economics, leetcode, pony, psychology, robotics, stackoverflow, sustainable-living, theoremqa-questions, theoremqa-theorems}` | nDCG@10 |
| `MSMARCOPassage` | `msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`, `msmarco-passage-v2/trec-dl-2022/judged` | nDCG@10, plus the dataset's official measures |
| `MSMARCODocument` | `msmarco-document/trec-dl-2019/judged`, `msmarco-document/trec-dl-2020/judged`, `msmarco-document-v2/trec-dl-2022/judged` | nDCG@10, plus the dataset's official measures |

Suite-specific behaviour:
- **BEIR:** queries use the `text` field. On Quora, results where `qid == docno` are removed. The 12
  CQADupStack forums are merged into one `beir/cqadupstack` row by geometric mean, but BEIR papers
  usually report the arithmetic mean.
- **BRIGHT:** a filter removes documents whose qrel label is `-100` (excluded from evaluation).
- **Lotte, MS MARCO:** measures from the dataset's documentation are added to the suite's own, and
  every extra measure appears in the Overall rows. Pass `eval_metrics=[...]` to control exactly what
  is reported.

## 4. Caching

| Argument | Layout | Reused when |
|---|---|---|
| `index_dir="indices"` | `indices/<corpus-id-slug>/`, given to the factory as `context.path` | your factory checks `index.built()` and skips indexing |
| `save_dir="runs"` | `runs/<dataset-id-slug>/<pipeline name>.res.gz` | a run file exists for the same dataset and pipeline name, and `save_mode` is not `"overwrite"` |

- The slug replaces `/` with `-` and lowercases, e.g. `nano-beir/scifact` becomes `nano-beir-scifact`.
- Pipeline names become file names: `pt.Experiment` rejects `:"%/<>^|?` and the path separator when
  `save_dir` is set. Use letters, digits and `.-_+`.
- Without `index_dir`, every corpus gets a new `tempfile.mkdtemp()` folder that is never deleted.
- A re-run with both caches was checked: it reused the index and the runs, and gave identical results.

## 5. Arguments

```python
Suite(ranking_generators, eval_metrics=None, subset=None, compute_overall=True, **experiment_kwargs)
```

- `subset` is a full dataset id as listed above, e.g. `"beir/fiqa/test"`, not `"fiqa"`.
- `compute_overall=False` drops the Overall rows. They are also skipped when `perquery=True`.
- `index_dir` and `save_dir` are used by the suite. Everything else (`baseline`, `correction`,
  `perquery`, `save_mode`, `batch_size`, ...) is passed to `pt.Experiment`.
- With a `baseline`, all pipelines of a corpus are kept in memory together for the significance test.
  Without one, they run one at a time.

## 6. Custom suites

```python
from suiteeval import Suite
from ir_measures import nDCG

MySuite = Suite.register(
    "my-suite",
    datasets=["beir/scifact/test", "beir/nfcorpus/test"],
    metadata={"official_measures": [nDCG@10], "description": "Two small BEIR datasets"},
)
results = MySuite(pipelines, eval_metrics=[nDCG@10])
```

For finer control, subclass the private suite class (e.g. `from suiteeval.suite.beir import _BEIR`)
and override hooks such as `wrap_pipeline`, `postprocess_results`, `measures_for` or `run_file_path`.

## 7. Known problems (0.1.8)

- The README example imports `NanoBEIR` but calls `BEIR(pipelines)`. Call the suite you imported.
- The paper's example uses `context.path / "index.pisa"`. `context.path` is a str, so use `os.path.join`.
- The README says temporary indexes are cleaned up. They are not.
- A run found in `save_dir` is replayed from its `.res.gz`: the pipeline doesn't execute, so its caches
  and `Lazy` models don't run and print no hit/miss line. Replayed runs are also evaluated without your
  `pt.Experiment` keyword arguments, so with `perquery=True` or `baseline=`, cached and fresh pipelines can
  be reported inconsistently. Use `save_mode="overwrite"` to test a cache, and for per-query results or
  significance tests.
- Importing `suiteeval` reads ir_datasets documentation for some suites, so the first import can be slow.
