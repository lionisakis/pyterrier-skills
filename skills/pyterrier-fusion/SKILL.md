---
name: pyterrier-fusion
description: >-
  Combines two or more PyTerrier rankings into one: reciprocal rank fusion
  (pyterrier-alpha RRFusion and rr_fusion), weighted score interpolation with
  per-query min-max normalisation and the + operator, set union and
  concatenation, fusing sparse with dense retrievers, fusing several rerankers,
  fusing saved runs, and tuning k or the weights. Use when a task says fuse,
  hybrid, combine, interpolate, ensemble or RRF, or when two systems' results
  should become one ranking. Not for building the retrievers (pyterrier-retrieval,
  pyterrier-splade, pyterrier-dense, pyterrier-colbert), for feature unions and
  learning to rank (pyterrier-reranking), or for evaluation (pyterrier-experiments).
compatibility: pyterrier>=1.1 (checked with 1.1.2 and pyterrier-alpha 0.16.5, CPU only)
---

# Fusing rankings

Assumes the `pyterrier-basics` skill.

Docs: [pyterrier-alpha](https://github.com/seanmacavaney/pyterrier-alpha) ·
[Operators](https://pyterrier.readthedocs.io/en/latest/operators.html) ·
Paper: Cormack, Clarke and Buettcher, Reciprocal rank fusion outperforms Condorcet and individual rank
learning methods (SIGIR 2009)

```bash
pip install pyterrier-alpha
```

## Choosing a fusion

| Need | Use | Combines | Notes |
|---|---|---|---|
| Merge rankings whose scores aren't comparable (the usual case) | `pta.RRFusion(a, b, k=60)` | `1 / (k + rank)` summed over the inputs | scale-free; the default choice |
| Weighted interpolation of scores | `(a >> norm) + w * (b >> norm)` with a per-query min-max `norm` | normalised scores, added | `w` is tuned on training topics |
| Fuse runs that already exist (saved runs, reranker outputs) | `pta.fusion.rr_fusion(run_a, run_b, k=60)` | as RRF, on result frames | needs a `query` column in every frame |
| A candidate set to rerank afterwards | `(a \| b) >> reranker` | the union of documents, scores dropped | `a ^ b` appends b's extra results after a's |
| Learn the combination from labels | `a ** b` and a learner | a `features` column | learning to rank; see the `pyterrier-reranking` skill |

Any transformers can be fused: two first stages, a first stage and a reranked list, or several rerankers
over the same candidates. The inputs must see the **same query text**.

## Reciprocal rank fusion

```python run
import pyterrier as pt
import pyterrier_alpha as pta

index = pt.terrier.TerrierIndex.example()
bm25, dph = index.bm25(), index.dph()

rrf = pta.RRFusion(bm25, dph, k=60)       # runs every input on the queries, then fuses
fused = rrf.search("chemical reactions")
assert {"qid", "query", "docno", "score", "rank"} <= set(fused.columns)
assert fused["rank"].iloc[0] == 0
```

- A document's fused score is the sum of `1 / (k + rank)` over the inputs that retrieved it (ranks from 0).
  `k=60` is the paper's value; a larger `k` flattens the difference between ranks.
- `num_results=1000` cuts the fused list per query. Pass `num_results=None` to keep every document.
- `RRFusion` runs each input pipeline itself. Two inputs that share a first stage run it twice: put the
  shared stage in a `RetrieverCache` (the `pyterrier-caching` skill), or fuse frames with `rr_fusion`.
- `k` is an attribute, so `pt.GridSearch(rrf, {rrf: {"k": [10, 60, 200]}}, train_topics, train_qrels,
  metric="map")` tunes it (see the `pyterrier-experiments` skill).

## Fusing result frames

```python run
run_a = bm25.search("chemical reactions")
run_b = dph.search("chemical reactions")
fused_frames = pta.fusion.rr_fusion(run_a, run_b, k=60)
assert fused_frames["docno"].tolist() == fused["docno"].tolist()    # same as RRFusion
```

`rr_fusion` merges on `qid`, `query` and `docno`, so every frame needs a `query` column with the same text.
A run read back from disk has none:

```python run
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "bm25.res.gz")
pt.io.write_results(run_a, path)
saved = pt.io.read_results(path)                     # qid, docno, rank, score, name: no query
topics = pt.new.queries(["chemical reactions"], qid=["1"])
saved = saved.merge(topics, on="qid")                # add the query text back
assert len(pta.fusion.rr_fusion(saved, run_b)) == len(fused_frames)
```

## Score interpolation

`+` adds scores and gives 0 to a document missing from one side, so normalise each side per query first:

```python run
def minmax(df):                                       # per query: scores to 0..1
    scores = df["score"]
    span = scores.max() - scores.min()
    return df.assign(score=(scores - scores.min()) / span if span > 0 else 0.0)

norm = pt.apply.by_query(minmax)
interpolated = (bm25 >> norm) + 0.5 * (dph >> norm)   # weight tuned on training topics
res = interpolated.search("chemical reactions")
assert 0 <= res["score"].min() and res["score"].max() <= 1.5
```

- pyterrier-alpha ships `pta.fusion.PerQueryMaxMinScore` for this, but in 0.16.5 (the latest release, and
  GitHub master) its `transform` fails with `ImportError: cannot import name 'validate' from
  'pyterrier_alpha.validate'`. Use the `pt.apply.by_query` normaliser above until a release fixes it.
- A query with one result gets score 0 from min-max. Fine for fusion; don't read the normalised scores alone.
- Tune the weight on training topics by comparing candidate weights in one `pt.Experiment`, never on the
  topics you report.

## Fusing rerankers

```python
from pyterrier_caching import RetrieverCache

candidates = RetrieverCache("bm25-top100.cache", bm25 % 100)      # the shared first stage, run once
ce = candidates >> dataset.text_loader() >> cross_encoder          # rerankers: the pyterrier-reranking skill
mono = candidates >> dataset.text_loader() >> mono_t5
ensemble = pta.RRFusion(ce, mono, k=60)
```

When `ce`, `mono` and `ensemble` are all evaluated, `RRFusion` runs each reranker again, so every
reranker scores the candidates once per system that contains it. Wrap expensive rerankers in a
`ScorerCache` (the `pyterrier-caching` skill) to score each pair once.

Rerankers keep `query` unchanged, so their outputs fuse directly. Runs produced by a pipeline with a query
rewriter (RM3, Bo1, SDM) carry the rewritten query: put `pt.rewrite.reset()` before the fusion, or the merge
treats the same document under two query strings as two documents.

## Evaluating a fusion

Compare every input and the fused system in one `pt.Experiment`, with the strongest input as baseline, and
report the method, `k` or the weights, and the depth of each input:

```python run
from pyterrier.measures import *

dataset = pt.get_dataset("irds:vaswani")
topics, qrels = dataset.get_topics(), dataset.get_qrels()
results = pt.Experiment(
    {"BM25": bm25, "DPH": dph, "RRF(BM25, DPH)": rrf, "0.5-interpolation": interpolated},
    topics, qrels, eval_metrics=[nDCG@10, AP], baseline="BM25", correction="bonferroni", round=4,
)
assert len(results) == 4
```

`pt.Experiment` may print a "could not be validated" warning for pipelines built with `+` or `pta.RRFusion`
(whether it does depends on the inputs). The scores are correct; `validate="ignore"` hides the warning.

On vaswani, RRF of BM25 and DPH moved nDCG@10 from 0.4085 (BM25) to 0.4264 on the first 20 topics: a
check that the code works, not a result to report.

## Don't

- Don't add raw scores from different systems (`bm25 + dense`). Their scales differ; fuse by rank or normalise first.
- Don't fuse pipelines whose `query` column differs (one after RM3, one not) without `pt.rewrite.reset()`.
- Don't pass frames without a `query` column to `rr_fusion`. Merge the topics in first.
- Don't use `pta.fusion.PerQueryMaxMinScore` with pyterrier-alpha 0.16.5. It raises `ImportError`.
- Don't use `|` or `&` when you need scores. They return documents only.
- Don't fuse inputs cut at different depths without saying so. The depth of each input is part of the method.
- Don't compare fused scores across runs or systems. Only the fused ranking is meaningful.
- Don't tune `k` or the weights on the topics you report.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: "['query'] not in index"` from `rr_fusion` | a frame has no `query` column (e.g. read with `pt.io.read_results`) | `run.merge(topics, on="qid")` first |
| Fused list has nearly twice the documents, same docno twice per query | the inputs carry different `query` text for the same `qid` | `pt.rewrite.reset()` before fusing |
| `ImportError: cannot import name 'validate' from 'pyterrier_alpha.validate'` | `PerQueryMaxMinScore` in pyterrier-alpha 0.16.5 | the `pt.apply.by_query` min-max above |
| Fused results stop at 1000 per query | `num_results=1000` default | `RRFusion(..., num_results=None)` |
| A shared first stage runs once per input | `RRFusion` calls each input pipeline | `RetrieverCache` around the shared stage, or `rr_fusion` on frames |
| `InputValidationError … score` after `\|` or `&` | set operators drop scores | rescore after the set operation, or use `RRFusion` |
| Fusion scores worse than its best input | inputs very unequal in quality, or one cut too shallow | weight the interpolation, or fuse at matching depths |
