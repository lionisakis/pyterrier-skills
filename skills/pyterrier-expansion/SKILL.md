---
name: pyterrier-expansion
description: >-
  Expands queries and documents in PyTerrier: pseudo-relevance feedback and
  query expansion (RM3, Bo1, KL), proximity rewriting with the sequential
  dependence model (SDM), interpolating rewritten and original queries, and index-time document expansion with doc2query
  and Doc2Query-- (generating queries for each document, scoring and filtering
  them). Use when adding query expansion, PRF, query rewriting or document
  expansion to a retrieval pipeline, or when choosing between them. Not for
  first-stage indexing in general, reranking or evaluation. For ColBERT PRF, use pyterrier-colbert.
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-doc2query 0.2.0 and GitHub master)
---

# Query and document expansion

Assumes the `pyterrier-basics` skill.

Docs: [Query rewriting](https://pyterrier.readthedocs.io/en/latest/terrier/rewrite.html) ·
[pyterrier-doc2query](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-doc2query/index.html)

## Choosing an expansion

| Goal | Use | When it runs | Needs |
|---|---|---|---|
| Add terms from top-ranked documents | RM3, Bo1 or KL (pseudo-relevance feedback) | query time | a Terrier index with a direct index (most have one) |
| Reward query terms that appear close together | SDM | query time | a Terrier index built with `store_positions=True` |
| PRF for ColBERT | PLAID-PRF | query time | see the `pyterrier-colbert` skill |
| Add likely queries to each document | doc2query, Doc2Query-- | index time | `pyterrier-doc2query` (+ `pyterrier-dr` for Doc2Query--); a GPU for full corpora, CPU is fine for a slice |

Query expansion changes the `query` column. Document expansion changes the documents before indexing.

## Pseudo-relevance feedback (RM3, Bo1, KL)

```python run
import pyterrier as pt

index = pt.terrier.TerrierIndex.example()
bm25 = index.bm25()

rm3 = (
    bm25                                           # 1. first retrieval
    >> index.rm3(fb_terms=10, fb_docs=3)           # 2. rewrite the query from the top fb_docs
    >> bm25                                        # 3. retrieve again with the expanded query
    >> pt.rewrite.reset()                          # 4. restore the original query for later stages
)
results = rm3.search("chemical reactions")
assert results["query"].iloc[0] == "chemical reactions"
```

- The shape is always **retrieve → expand → retrieve**. The expansion step needs the first
  retrieval's results.
- The rewritten query is a weighted Terrier query (`applypipeline:off chemic^0.338 reaction^0.393 …`),
  and the original is kept in `query_0`. Run `pt.rewrite.reset()` before any neural reranker, so it
  sees the user's query.
- `index.rm3()`, `index.bo1()` and `index.kl()` take `fb_terms` and `fb_docs` (RM3 also takes
  `fb_lambda`). The class forms are `pt.rewrite.RM3(index, ...)`, `Bo1QueryExpansion`,
  `KLQueryExpansion`.
- Gains are collection-dependent and often small. On vaswani, MAP was 0.2965 for BM25, 0.2984 with RM3,
  0.3046 with Bo1 and 0.3025 with KL. Tune `fb_terms` and `fb_docs` on training topics (see the
  `pyterrier-experiments` skill).

## Sequential dependence model (SDM)

```python run
dataset = pt.get_dataset("irds:vaswani")
positional = pt.terrier.TerrierIndex("vaswani.positions.terrier")
if not positional.built():
    positional.indexer(meta={"docno": 20}, store_positions=True).index(dataset.get_corpus_iter())

sdm = positional.sdm() >> positional.bm25()
assert "#uw8" in sdm.search("chemical reactions")["query"].iloc[0]   # proximity operators added
```

- SDM needs term positions. On an index built without them it fails with
  `AssertionError: … missing positional information`. Prebuilt indexes rarely store positions.
- SDM rewrites the query into exact-phrase (`#1`) and unordered-window (`#uw8`) operators. On vaswani it
  moved MAP from 0.2965 to 0.2981.

## Mixing the expanded and original query

`pt.rewrite.linear(w_current, w_previous)` combines the latest rewrite with the previous query:

```python
mixed = bm25 >> index.rm3() >> pt.rewrite.linear(0.7, 0.3) >> bm25
```

## Document expansion with doc2query

Expansion happens **before indexing**: generate queries for each document, optionally filter them,
then index the result.

```python
from pyterrier_doc2query import Doc2Query

doc2query = Doc2Query(num_samples=40, batch_size=16, append=True)    # append=True adds queries to `text`
index = pt.terrier.TerrierIndex("collection.d2q.terrier")
(doc2query >> index.indexer(meta={"docno": 20})).index(dataset.get_corpus_iter())
```

- By default (`append=False`; the checkpoint and other defaults are in
  [references/doc2query.md](references/doc2query.md)) the queries go into a `querygen` column, which a
  plain indexer ignores. Set `append=True`, or filter first (Doc2Query--, below).
- Generation is slow: one T5 generation per document and sample. Run it once and keep the output with
  `IndexerCache` (see the `pyterrier-caching` skill). Prebuilt doc2query indexes for MS MARCO exist,
  e.g. `pt.Artifact.from_dataset("msmarco_passage", "terrier_stemmed_docT5query")` (see the
  `pyterrier-huggingface` skill).

For Doc2Query-- (scoring generated queries with a relevance model and dropping the worst), read
[references/doc2query.md](references/doc2query.md).

## Don't

- Don't put query expansion first in a pipeline. It needs results from a first retrieval.
- Don't forget `pt.rewrite.reset()` before a neural reranker or anything that reads `query`.
- Don't use SDM on an index without positions.
- Don't index doc2query output with `append=False` and no filter step. The queries would be dropped.
- Don't regenerate doc2query queries for every experiment. Cache them.
- Don't use `pt.rewrite.stash_results()` before a query expansion step. It hides the results the
  expansion needs (`InputValidationError`).

## Reference files

| Read | When |
|---|---|
| [references/doc2query.md](references/doc2query.md) | generating, scoring, filtering or caching doc2query expansions |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: … missing positional information` | SDM on an index without positions | rebuild with `store_positions=True` |
| A neural reranker scores poorly after RM3 | it received the weighted Terrier query | `pt.rewrite.reset()` before the reranker |
| `InputValidationError` naming `stashed_results_0` | `stash_results()` placed before an expansion step | remove it |
| Expansion changes nothing | `fb_docs` too small, or the first stage returns few results | check the first stage's results; tune `fb_docs` and `fb_terms` |
| Indexed doc2query documents look unexpanded | `append=False` without a filter step | `Doc2Query(append=True)` or add `QueryFilter(..., append=True)` |
| `ValueError: Error parsing line b'\x0e' in …/spiece.model` when creating `Doc2Query` | transformers 5 without `protobuf` | `pip install protobuf` |
| `KeyError: 'docno'` from `QueryScorer` | bug in pyterrier-doc2query 0.2.0 on PyPI | install from GitHub; see references/doc2query.md |
