# Learning to rank

Checked with PyTerrier 1.1.2, scikit-learn and LightGBM, on vaswani.

## Contents
1. Building features
2. Training and applying a model
3. Feature selection and ablation
4. Method

## 1. Building features

Features are a numpy array in the `features` column, one entry per feature transformer.

| Way | Example | Notes |
|---|---|---|
| Feature union `**` | `bm25 % 100 >> (index.retriever("Tf") ** index.pl2() ** index.dph())` | any transformers that score documents; the first stage's `score` is kept |
| Terrier `FeaturesRetriever` | `pt.terrier.FeaturesRetriever(index, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])` | computes Terrier features in one pass; faster on large indexes |
| Your own feature | `pt.apply.doc_score(fn)` inside `**`, or `pt.apply.doc_features(fn)` returning an array | e.g. document length, a neural score |
| The current score as a feature | `pt.ltr.score_to_feature()` | appends `score` to `features` |

```python
features = bm25 % 100 >> (
    index.retriever("Tf")
    ** index.pl2()
    ** pt.apply.doc_score(lambda row: len(row["text"]), label="doclen")   # needs text loaded first
)
```

Check the features before training: `features.search("query")["features"].iloc[0]`.

## 2. Training and applying a model

| Learner | Call |
|---|---|
| scikit-learn regressor or classifier (e.g. `RandomForestRegressor`) | `pt.ltr.apply_learned_model(learner)` (pointwise, `form="regression"`) |
| LightGBM `LGBMRanker`, XGBoost `XGBRanker` | `pt.ltr.apply_learned_model(learner, form="ltr")` (listwise, grouped by query) |
| FastRank | `pt.ltr.apply_learned_model(learner, form="fastrank")` |

```python
import lightgbm as lgb

ranker = features >> pt.ltr.apply_learned_model(
    lgb.LGBMRanker(n_estimators=100, verbose=-1), form="ltr"
)
ranker.fit(train_topics, qrels, valid_topics, qrels)
```

- `fit(train_topics, train_qrels, valid_topics, valid_qrels)` runs the feature pipeline on the topics,
  labels the results from the qrels (documents without a judgement get label 0), and trains.
- With `form="regression"` (scikit-learn), the validation topics are **ignored**.
- With `form="ltr"`, they are **required** (`ValueError: No validation results to fit to`) and are
  passed to the learner as `eval_set` and `eval_group`. That is LightGBM's interface, and what its
  early stopping uses.
- After `fit`, the pipeline ranks by the model's prediction.
- A learner fits one feature layout. After changing the features, create a new learner object.

## 3. Feature selection and ablation

| Need | Use |
|---|---|
| Drop features by position | `features >> pt.ltr.ablate_features([0, 2])` |
| Keep only some features | `features >> pt.ltr.keep_features([1])` |
| Rank by one feature | `features >> pt.ltr.feature_to_score(1)` |

Compare each ablation against the full model in one `pt.Experiment`.

## 4. Method

- Use separate training, validation and test topics. Report results only on the test topics.
- Compare against the first stage. With few training topics, learning to rank can be worse than
  BM25. On vaswani with 40 training topics, a random forest scored nDCG@10 0.225 and LightGBM 0.278,
  against BM25's 0.376 on the same test topics.
- Features from the same first stage are correlated. More features need more training topics.
- Normalise features if the learner is sensitive to scale (linear models, neural rankers). Tree
  models don't need it.
