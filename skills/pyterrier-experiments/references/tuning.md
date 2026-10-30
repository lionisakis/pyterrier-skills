# Tuning reference

Checked with PyTerrier 1.1.2 and Terrier 5.11.

## Contents
1. Signatures
2. Making a parameter tunable
3. Metrics
4. Cross-validation
5. Known problems

## 1. Signatures

```python
pt.GridSearch(pipeline, params, topics, qrels, metric="map", jobs=1, backend="joblib",
              verbose=False, batch_size=None, return_type="opt_pipeline")
pt.GridScan(pipeline, params, topics, qrels, metrics=["map"], jobs=1, backend="joblib",
            verbose=False, batch_size=None, dataframe=True)
pt.KFoldGridSearch(pipeline, params, topics_list, qrels, metric="map", jobs=1, backend="joblib",
                   verbose=False, batch_size=None)
```

- `params` maps each transformer *object* in the pipeline to `{parameter_name: [values]}`.
- `GridSearch` returns the pipeline set to its best values (`return_type="opt_pipeline"`), the best
  `(score, [(transformer, param, value), ...])` (`"best_setting"`; get the values with
  `{p: v for _, p, v in settings}`), or both. It also prints the best value and setting.
- `GridScan` returns a DataFrame with one row per combination, one column per metric and one column per
  parameter, named `tran_<i>_<param>` (e.g. `tran_0_bm25.b`). Use it to
  see how sensitive a pipeline is to a parameter. **It leaves the transformer set to the last value it
  tried**, so create the retriever again before using it for anything else. To get both the curve and
  the best value in one pass, run `GridScan` once and take the row with the highest score.
- Both average the metric over every qid in `qrels`, without filtering by `topics`. Pass
  `qrels[qrels.qid.isin(topics.qid)]` when tuning on a topic subset.
- `KFoldGridSearch` takes a list of topic folds. For each fold it tunes on the other folds and retrieves
  for the held-out fold. It returns `(results, settings)`. `results` is a results frame covering every
  topic, retrieved with each fold's tuned setting. Evaluate it like any run. `settings` holds the best
  setting found for each fold.

## 2. Making a parameter tunable

Terrier retrievers expose parameters through `controls`. A parameter can only be tuned if it was
given an initial value when the retriever was created:

```python
bm25 = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b": 0.75, "bm25.k_1": 1.2})
best = pt.GridSearch(
    bm25,
    {bm25: {"bm25.b": [0.3, 0.5, 0.75, 0.9], "bm25.k_1": [0.9, 1.2, 1.5]}},
    train_topics, qrels[qrels.qid.isin(train_topics.qid)], metric="map", return_type="best_setting",
)
```

- `TerrierIndex.retriever()` and `TerrierIndex.bm25()` don't take `controls=`. Use
  `pt.terrier.Retriever(...)` for anything you tune.
- In Terrier 5.11, the control `"c"` sets the same BM25 `b` parameter as `"bm25.b"` (checked: both give
  identical scores). Prefer `"bm25.b"`, because its meaning is clear.
- Other transformers expose their own parameters through `get_parameter`/`set_parameter`. A
  `pt.apply` function has no tunable parameters; make it a class with attributes instead.

## 3. Metrics

`GridSearch(metric=...)` and `GridScan(metrics=...)` look results up by the string you pass, so use
trec_eval-style names:

| Want | Pass |
|---|---|
| MAP | `"map"` |
| nDCG@10 | `"ndcg_cut_10"` |
| P@10 | `"P_10"` |
| Recall@1000 | `"recall_1000"` |
| MRR | `"recip_rank"` |

All five strings above were checked. Measure objects (`nDCG@10`, `AP`) raise `KeyError`, and
ir_measures strings such as `"nDCG@10"` raise `ValueError: unkonwn measure`.

When the measure needs `rel=` (e.g. `AP(rel=2)` on TREC DL), skip GridSearch. Build one retriever per
setting and compare them on the training topics in one experiment:

```python
from pyterrier.measures import *
candidates = {
    f"b={b}": pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b": b})
    for b in [0.3, 0.5, 0.75, 0.9]
}
scan = pt.Experiment(candidates, train_topics, qrels, eval_metrics=[AP(rel=2)])
best_name = scan.sort_values("AP(rel=2)", ascending=False).iloc[0]["name"]
```

## 4. Cross-validation

```python
folds = [topics.iloc[i::5] for i in range(5)]
results, settings = pt.KFoldGridSearch(
    bm25, {bm25: {"bm25.b": [0.3, 0.5, 0.75]}}, folds, qrels, metric="map"
)
scores = pt.Experiment([results], topics, qrels, eval_metrics=[AP], names=["BM25 (5-fold tuned)"])
```

Use cross-validation when the collection has no separate training topics (e.g. Robust04), and report
the per-fold settings.

## 5. Known problems

- The docs' BM25 tuning example has missing commas. Don't copy it.
- Tuning has no `save_dir`. Each combination re-runs the whole pipeline, so put an expensive first stage
  behind a cache (see the `pyterrier-caching` skill).
- `GridScan` doesn't restore the original parameter values (see section 1).
