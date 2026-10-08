---
name: pyterrier-reranking
description: >-
  Reranks PyTerrier first-stage results with neural and learned rerankers:
  cross-encoders (sentence-transformers CrossEncoder, monoELECTRA), monoT5 and
  duoT5, LLM listwise rerankers (RankZephyr, RankGPT, LiT5 via pyterrier-
  generative), passage-level scoring of long documents (sliding windows with
  max-passage aggregation), and learning to rank with feature unions and
  LightGBM or scikit-learn. Use when adding a reranker or second stage to a
  pipeline, choosing a rerank depth, scoring query-document pairs, or training a
  learning-to-rank model in PyTerrier. Not for first-stage retrieval, query
  expansion or evaluation. For SPLADE, dense or ColBERT as rerankers, use
  pyterrier-splade, pyterrier-dense or pyterrier-colbert.
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-t5 0.3.0, pyterrier-dr 0.8.0, sentence-transformers 6.1, lightgbm, CPU only)
---

# Reranking

Assumes the `pyterrier-basics` skill.

Docs: [Neural rerankers](https://pyterrier.readthedocs.io/en/latest/neural.html) ·
[Text and passages](https://pyterrier.readthedocs.io/en/latest/text.html) ·
[Learning to rank](https://pyterrier.readthedocs.io/en/latest/ltr.html) ·
[pyterrier_t5](https://github.com/terrierteam/pyterrier_t5) ·
[pyterrier-dr](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-dr/index.html) ·
[pyterrier-generative](https://github.com/Parry-Parry/pyterrier-generative)

## The shape of every reranking pipeline

```
first_stage % k  >>  text_loader  >>  reranker  [ % k2 >> second_reranker ]
```

1. **`% k`, the rerank depth.** It is a cost and quality decision. Take it from the task or the
   paper, state it in the report, and keep it the same across compared systems. 100–1000 is common
   for cross-encoders, and smaller for LLM rerankers.
2. **A text loader.** Rerankers read the `text` column, and most indexes don't return text. Use
   `dataset.text_loader()` or `index.text_loader()`.
3. **The reranker** replaces `score` and recomputes `rank`. Results the first stage didn't retrieve
   can't come back, so recall at `k` caps what the reranker can achieve.

```python run
import pyterrier as pt

dataset = pt.get_dataset("irds:vaswani")
index = pt.terrier.TerrierIndex.example()
bm25 = index.bm25(k1=1.2, b=0.75)

def overlap(df):  # a stand-in scorer: replace with a real model
    return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in zip(df["query"], df["text"])]

pipeline = bm25 % 20 >> dataset.text_loader() >> pt.apply.doc_score(overlap, batch_size=32)
reranked = pipeline.search("chemical reactions")
assert len(reranked) == 20 and reranked["rank"].min() == 0
```

## Choosing a reranker

| Reranker | Use | Install | Typical depth |
|---|---|---|---|
| Any `CrossEncoder` from sentence-transformers | quick, many checkpoints; via `pt.apply.doc_score` | `sentence-transformers` | 100–1000 |
| monoELECTRA (`pyterrier_dr.ElectraScorer()`) | strong passage cross-encoder | `pyterrier-dr` | 100–1000 |
| monoT5 (`pyterrier_t5.MonoT5ReRanker()`) | strong pointwise T5 reranker | `pyterrier-t5` | 100–1000 |
| duoT5 (`pyterrier_t5.DuoT5ReRanker()`) | pairwise, after monoT5 on a small top-k | `pyterrier-t5` | 10–50 (cost grows with k²) |
| ColBERT text scorer | late interaction without an index | `pyterrier_colbert2`; see the `pyterrier-colbert` skill | 100–1000 |
| Dense bi-encoder as scorer (`model` from pyterrier-dr) | cheap, reuses a dense model | `pyterrier-dr`; see the `pyterrier-dense` skill | 100–1000 |
| SPLADE scorer (`splade.scorer()`) | learned sparse, without an index | `pyterrier-splade`; see the `pyterrier-splade` skill | 100–1000 |
| LLM listwise (RankZephyr, RankGPT, LiT5) | strongest, most expensive; GPU or API key. **0.1.3 doesn't import with released dependencies; check first** | `pyterrier-generative` | 20–100 |
| Learning to rank | combine many hand-built features | core + `lightgbm` or scikit-learn | 100–1000 |

Read [references/neural.md](references/neural.md) before configuring a neural reranker,
[references/llm-rerankers.md](references/llm-rerankers.md) before using an LLM reranker, and
[references/ltr.md](references/ltr.md) before training a learning-to-rank model.

## Cross-encoders

```python run slow
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

def cross_encoder_score(df):
    return cross_encoder.predict(list(zip(df["query"], df["text"])))

ce_pipeline = bm25 % 3 >> dataset.text_loader() >> pt.apply.doc_score(
    cross_encoder_score, batch_size=32, label="MiniLM"
)
assert len(ce_pipeline.search("chemical reactions")) == 3
```

```python run slow
import pyterrier_dr

electra = pyterrier_dr.ElectraScorer(batch_size=16)     # crystina-z/monoELECTRA_LCE_nneg31
electra_pipeline = bm25 % 3 >> dataset.text_loader() >> electra
assert len(electra_pipeline.search("chemical reactions")) == 3
```

- Always score in batches (`batch_size=` in `pt.apply.doc_score`, or the reranker's own `batch_size`).
- Load the model once, outside the scoring function.

## monoT5 and duoT5

```python run slow
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

mono_t5 = MonoT5ReRanker(batch_size=4)       # castorini/monot5-base-msmarco
duo_t5 = DuoT5ReRanker(batch_size=4)         # castorini/duot5-base-msmarco
t5_pipeline = bm25 % 3 >> dataset.text_loader() >> mono_t5 % 2 >> duo_t5
assert len(t5_pipeline.search("chemical reactions")) == 2
```

- The classes are `MonoT5ReRanker` and `DuoT5ReRanker`. There is no `MonoT5` class.
- `text_field=` selects a column other than `text`. Inputs longer than the model's 512 tokens are
  truncated.
- duoT5 compares every pair, so cut down first (`mono_t5 % 10 >> duo_t5`). Only the top `k` survive.

## Long documents

Score passages and keep each document's best passage score:

```python run
import pandas as pd

long_doc = pd.DataFrame([{
    "qid": "1", "query": "chemical reactions", "docno": "d1", "score": 1.0, "rank": 0,
    "text": " ".join(["filler"] * 300) + " chemical reactions",
}])
passager = pt.text.sliding(text_attr="text", length=128, stride=64, prepend_attr=None)
passages = passager(long_doc)
assert passages["docno"].tolist() == ["d1%p0", "d1%p1", "d1%p2"]
```

```python
long_doc_pipeline = (
    first_stage % 100
    >> dataset.text_loader()
    >> pt.text.sliding(text_attr="text", length=128, stride=64, prepend_attr=None)
    >> reranker
    >> pt.text.max_passage()       # or first_passage(), mean_passage(), kmaxavg_passage(k)
)
```

`pt.text.sliding()` defaults to `text_attr="body"` and `prepend_attr="title"`. On data with a
`text` column and no `title`, pass both arguments, as above, or it fails with `InputValidationError`.

## Learning to rank

```python run
from sklearn.ensemble import RandomForestRegressor

topics, qrels = dataset.get_topics(), dataset.get_qrels()
train, valid = topics.iloc[:40], topics.iloc[40:60]

features = bm25 % 50 >> (index.retriever("Tf") ** index.pl2() ** index.dph())
ltr = features >> pt.ltr.apply_learned_model(RandomForestRegressor(n_estimators=20))
ltr.fit(train, qrels, valid, qrels)
assert len(ltr.search("chemical reactions")) == 50
```

- `**` builds a `features` array, one entry per transformer, while the first stage keeps the score.
- For LightGBM, XGBoost or FastRank rankers, pass `form="ltr"` or `form="fastrank"`. Read
  [references/ltr.md](references/ltr.md).
- Train, validate and test on separate topics. Compare against the first stage: with few training
  topics, a learned model can score **below** BM25 (on 40 vaswani training topics, nDCG@10 fell from
  0.376 to 0.225).

## Evaluating a reranker

Evaluate the first stage and every reranked pipeline in one `pt.Experiment`, with the first stage as
baseline (see the `pyterrier-experiments` skill):

```python
pt.Experiment(
    [bm25 % k, bm25 % k >> dataset.text_loader() >> reranker],
    topics, qrels,
    eval_metrics=[nDCG@10],
    names=["BM25", "BM25 >> reranker"],
    baseline=0, correction="holm",
)
```

To run several rerankers or evaluations over the same candidates, cache the scores (see the
`pyterrier-caching` skill).

## Don't

- Don't put a reranker after a first stage without a text loader.
- Don't rerank the full first-stage list by accident. Always write `% k` and choose `k` deliberately.
- Don't use `pyterrier_t5.MonoT5`. The class is `MonoT5ReRanker`.
- Don't run duoT5 on more than a small top-k.
- Don't call a neural model once per row. Batch.
- Don't use `pt.text.sliding()` with its defaults on `text` data.
- Don't train learning to rank on the topics you report, or reuse a learner after the feature
  count changes.
- Don't plan around pyterrier-generative before checking that `import pyterrier_generative` works, and
  don't copy its README code as-is. It uses `pt.init()` and `pt.BatchRetrieve`.

## Reference files

| Read | When |
|---|---|
| [references/neural.md](references/neural.md) | configuring cross-encoders, monoT5/duoT5, or monoELECTRA |
| [references/llm-rerankers.md](references/llm-rerankers.md) | using RankZephyr, RankVicuna, RankGPT, LiT5 or a custom LLM reranker |
| [references/ltr.md](references/ltr.md) | building features and training learning-to-rank models |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `InputValidationError … missing_columns=['text']` | no text loader before the reranker | add `dataset.text_loader()` |
| `AttributeError: module 'pyterrier_t5' has no attribute 'MonoT5'` | wrong class name | `MonoT5ReRanker` |
| `InputValidationError` from `SlidingWindowPassager` | `sliding()` defaults expect `body` and `title` | `pt.text.sliding(text_attr="text", prepend_attr=None)` |
| Reranking is extremely slow | CPU, depth too large, or no batching | GPU; smaller `k`; set `batch_size` |
| Reranked scores worse than the first stage | domain or text mismatch (e.g. titles only), too-small depth, or an undertrained LTR model | check the text column; try a larger `k`; compare on validation topics |
| `ValueError: Model … expected 2 features, but found 3 features for inference` | a learner trained on a different feature set | create a new learner after changing features |
| `ModuleNotFoundError: No module named 'pyterrier_rag.prompt.jinja'` on `import pyterrier_generative` | pyterrier-generative 0.1.3 needs an unreleased pyterrier-rag module | use a cross-encoder or monoT5; see references/llm-rerankers.md |
| Results disappear after duoT5 | the `% k` before duoT5 limits the output | keep the remaining results with `^`, e.g. `(mono % 10 >> duo) ^ mono` |
