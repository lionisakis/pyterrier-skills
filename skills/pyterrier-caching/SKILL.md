---
name: pyterrier-caching
description: >-
  Caches expensive parts of PyTerrier pipelines with pyterrier-caching so
  repeated runs, experiments and parameter sweeps don't recompute them: reranker
  and scorer scores (ScorerCache, DenseScorerCache), first-stage results
  (RetrieverCache), document encodings and other indexing-pipeline output such
  as SPLADE or doc2query (IndexerCache), and any deterministic column-to-column
  transform (KeyValueCache), plus Lazy model loading and prebuilt caches from
  Hugging Face. Use when a pipeline stage is slow and will run more than once,
  when replacing the removed ~ caching operator, or when loading a cached
  monoELECTRA or SPLADE artifact. Not for choosing rerankers or retrievers.
compatibility: pyterrier>=1.1 (checked with 1.1.2 and pyterrier-caching 0.5.8)
---

# Caching pipeline stages

Assumes the `pyterrier-basics` skill.

Docs: [pyterrier-caching](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-caching/index.html) ·
[GitHub](https://github.com/seanmacavaney/pyterrier-caching)

```bash
pip install pyterrier-caching
```

This package replaces the `~` caching operator, which was removed in PyTerrier 1.0.

## Choosing a cache

| Expensive stage | Cache | Keyed on | Stores |
|---|---|---|---|
| Reranker / scorer over a sample of documents | `ScorerCache(path, scorer)` (SQLite, sparse) | query **text** + `docno` | `score` only |
| Scorer over most or all of a corpus | `DenseScorerCache(path, scorer)` (HDF5) | query text + document position | `score` only |
| First-stage retriever | `RetrieverCache(path, retriever)` | all input columns, or `on=[...]` | the full results |
| Indexing-pipeline output (encodings, expansions) | `IndexerCache(path)` | document order | every field of every document |
| Any deterministic transform | `KeyValueCache(path, transformer, key=..., value=...)` | the `key` columns | the `value` columns |

All of them are transformers (or an indexer, for `IndexerCache`), so they drop into a pipeline where
the stage they wrap used to be.

## Scorer cache

```python run
import pyterrier as pt
from pyterrier_caching import ScorerCache

dataset = pt.get_dataset("irds:vaswani")
index = pt.terrier.TerrierIndex.example()
bm25 = index.bm25()

calls = {"docs": 0}
def expensive_score(df):              # stands in for a neural reranker
    calls["docs"] += len(df)
    return df["text"].str.len()

reranker = dataset.text_loader() >> pt.apply.doc_score(expensive_score, batch_size=64)
cached = ScorerCache("vaswani.length-scorer.cache", reranker)    # one cache per scorer and corpus

pipeline = bm25 % 10 >> cached
first = pipeline.search("chemical reactions")
second = pipeline.search("chemical reactions")
assert calls["docs"] == 10 and first["score"].tolist() == second["score"].tolist()
```

- Put the text loader **inside** the cached transformer, as above. On a hit, text isn't loaded at all.
- A scorer built eagerly (`CrossEncoder(...)` at the top of the script) loads its model even when every
  pair is a hit. Wrap it in `InspectableLazy` (below) when re-runs should be fast.
- The cache stores only `score`. Columns the scorer adds (e.g. `text`) aren't in the output.
- The key is the query **text** and `docno`, not `qid`. The same query under another `qid` is a hit.
- The cache doesn't know which model produced a score. **A different scorer given the same path
  silently returns the old scores.** Name the path after the scorer and corpus, and delete it when
  the model, its settings or the text change.
- Pass `verbose=True` to see what hit: each call prints `<cache>: N hit(s), M miss(es)`.
- `Lazy(factory)` delays loading a model, but in pyterrier-caching 0.5.8 `ScorerCache` inspects its
  scorer on every call, and that loads a plain `Lazy` even on a 100% hit. Declare its columns, as below.

```python
from pyterrier_caching import Lazy, ScorerCache

class InspectableLazy(Lazy):
    def transform_inputs(self):          # needed when nothing comes before it in the cached pipeline
        return [["qid", "query", "docno", "text"]]

    def transform_outputs(self, input_columns):    # needed after a text loader
        return list(input_columns)

scorer = ScorerCache(path, dataset.text_loader() >> InspectableLazy(make_model), verbose=True)
```

`make_model` can be a class or any function that returns a transformer. Read
[references/scorer-caches.md](references/scorer-caches.md).

## Retriever cache

```python run
from pyterrier_caching import RetrieverCache

cached_bm25 = RetrieverCache("vaswani.bm25.cache", bm25)
topics = dataset.get_topics().head(5)
fresh = cached_bm25(topics)                 # runs BM25 and stores the results
cached = cached_bm25(topics)                # read from disk (about 40x faster here)
columns = ["qid", "docno", "rank", "score"]
assert fresh[columns].reset_index(drop=True).equals(cached[columns].reset_index(drop=True))
```

- By default every input column is part of the key, so a query frame with an extra column misses. Use
  `on="query"` (or a list) to key on chosen columns.
- The rows come back with a different DataFrame index, so compare cached and fresh results on their
  columns, not with `.equals()` directly.
- The cache is a **directory** at `path` (`<hash>.dbm.dat`, `<hash>.dbm.dir`, `<hash>.dbm.bak` and
  `pt_meta.json`). Delete it with `shutil.rmtree(path)`; `os.remove(path)` raises `IsADirectoryError`.
- The files use `dbm.dumb`, which has no locking. Don't share one cache between processes running at
  the same time.

## Indexer cache

Run an expensive indexing pipeline once (e.g. SPLADE encoding, doc2query expansion), store its
output, then index it as many times as you like:

```python
from pyterrier_caching import IndexerCache

cache = IndexerCache("corpus.splade.cache")
(splade.doc_encoder() >> cache).index(dataset.get_corpus_iter())      # expensive, once

terrier = pt.terrier.TerrierIndex("corpus.splade.terrier")
terrier.toks_indexer().index(iter(cache))                             # cheap, repeatable
```

`iter(cache)` (or `cache.get_corpus_iter()`) yields the stored documents; `cache.to_dataframe()` loads
them into a frame. Check `cache.built()` first: `len()` and iteration on an unbuilt cache raise
`RuntimeError: cache not built`. `cache.text_loader()` loads
stored fields into results, but only when **every** docno in the results is in the cache. A missing
docno fails with the unhelpful `ValueError: not enough values to unpack (expected 2, got 0)`.

## Key-value cache

```python
from pyterrier_caching import KeyValueCache

cached_rewriter = KeyValueCache("query-rewrites.cache", llm_rewriter, key="query", value="query")
```

The transformer runs only for keys that aren't stored yet. Use it for query rewriting, query
generation, or any other column computed from other columns.

## Prebuilt caches on Hugging Face

Some caches are published as artifacts, e.g. `macavaney/msmarco-passage.monoelectra.cache` and the
BEIR `pyterrier/<corpus>.splade-v3.cache` indexer caches. Load them with `pt.Artifact.from_hf(...)`,
which returns the right cache class, and check sizes first (see the `pyterrier-huggingface` skill).

## Don't

- Don't use `~transformer`. Use one of these caches.
- Don't reuse a cache path for a different model, settings, text or corpus. Old scores come back
  silently.
- Don't load text outside the cached scorer. That loads text even on hits.
- Don't share a `RetrieverCache` between concurrent processes.
- Don't call `cached_retriever()` on a dense scorer cache unless every document has been scored.
- Don't cache transformers whose output changes between calls (sampling, time-dependent APIs)
  without accepting that the first answer is kept.

## Reference files

| Read | When |
|---|---|
| [references/scorer-caches.md](references/scorer-caches.md) | choosing sparse or dense scorer caches, `Lazy`, scoring a whole corpus, merging caches, or loading a prebuilt score cache |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Scores don't change after changing the model | same cache path for a different scorer | new path, or delete the old cache |
| `RuntimeError: retriever only works if corpus is scored completely` | `cached_retriever()` on a partly filled dense cache | `cache.score_all(dataset)` first, or use the cache as a scorer |
| Cached retriever always misses | input frames differ in some column | `RetrieverCache(path, retriever, on="query")` |
| The model loads even when everything is cached (`verbose=True` shows 0 misses) | the model is built eagerly, or a plain `Lazy` is loaded by the cache's input inspection | wrap it in a `Lazy` subclass that defines `transform_outputs` (see "Scorer cache") |
| `KeyError: 'text'` after a scorer cache | the cache outputs `score`, not the scorer's extra columns | load text after the cache if a later stage needs it |
| `ValueError: No implementation found … pip install pyterrier-caching` when loading a `.cache` artifact | package not installed | `pip install pyterrier-caching` |
| `ValueError: not enough values to unpack (expected 2, got 0)` from `IndexerCache.text_loader()` | a result docno that isn't in the indexer cache | cache the whole corpus, or load text from the dataset |
| `RuntimeError: cache not built` from `len(cache)` or `iter(cache)` | the `IndexerCache` has not been filled yet | test `cache.built()` before reading it |
| `AssertionError: unknown docno encountered` from a dense cache | a docno that isn't in the corpus the cache was built from | build the cache from the same corpus the first stage retrieves from |
| `IsADirectoryError: [Errno 21] Is a directory: <cache path>` | `os.remove` on a cache path; every cache is a directory | `shutil.rmtree(path)` |
