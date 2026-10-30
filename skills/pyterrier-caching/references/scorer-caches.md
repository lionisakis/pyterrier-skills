# Scorer caches

Checked with pyterrier-caching 0.5.8 and PyTerrier 1.1.2.

## Contents
1. Sparse or dense
2. Avoiding model loading with Lazy
3. Scoring a whole corpus
4. Merging and sharing caches

## 1. Sparse or dense

| | `ScorerCache` = `SparseScorerCache` = `Sqlite3ScorerCache` | `DenseScorerCache` = `Hdf5ScorerCache` |
|---|---|---|
| Storage | SQLite, one row per (query, docno) scored | HDF5, one float array of corpus size per query |
| Setup | none | `cache.build(dataset.get_corpus_iter())` once, to record the corpus docnos |
| Best for | reranking a top-k sample of documents | scoring most or all of a corpus |
| Key | query text + `docno` (`group=`, `key=`, `value=` to change the columns) | query text + docno position in the built corpus |
| Extra | `pickle=True` stores non-float values | `cached_retriever(num_results)` ranks the whole corpus once every document is scored |

```python
from pyterrier_caching import ScorerCache, DenseScorerCache

sparse = ScorerCache("msmarco.monoelectra.sqlite", dataset.text_loader() >> scorer)

dense = DenseScorerCache("msmarco.monoelectra.hdf5", dataset.text_loader() >> scorer)
dense.build(dataset.get_corpus_iter())         # once
pipeline = bm25 % 100 >> dense
```

Both caches are keyed on query **text**. Two topic sets that share query strings share cache entries,
which is usually what you want.

## 2. Avoiding model loading with Lazy

```python
from pyterrier_caching import Lazy, ScorerCache
from pyterrier_dr import ElectraScorer

class InspectableLazy(Lazy):
    def transform_outputs(self, input_columns):    # lets inspection skip loading the model
        return list(input_columns)

scorer = ScorerCache("electra.cache", dataset.text_loader() >> InspectableLazy(ElectraScorer))
```

`Lazy(cls, *args, **kwargs)` creates the transformer on its first real call. Don't use a plain `Lazy`
here: in pyterrier-caching 0.5.8 `ScorerCache` inspects its scorer on every call, and inspecting a
`text_loader() >> Lazy(...)` pipeline loads the model even on a 100% hit. With `transform_outputs`
declared, a fully cached experiment never loads the model or touches the GPU (see "Scorer cache" in
SKILL.md, which also covers a cached pipeline with nothing before the `Lazy`).

## 3. Scoring a whole corpus

```python
dense.score_all(dataset, batch_size=1024)   # every topic against every document
exhaustive = dense.cached_retriever(num_results=1000)
```

`score_all` scores the dataset's topics against the whole corpus, which is only practical for small
corpora or cheap scorers. `cached_retriever` raises
`RuntimeError: retriever only works if corpus is scored completely` until that is done.

## 4. Merging and sharing caches

- `cache.merge_from(other)` copies another cache of the same kind into this one. On overlap, entries
  from `other` win. Use it to combine caches filled on different machines.
- A cache folder holds `pt_meta.json`, so it is an artifact. Share it with `cache.to_hf(...)` and load it
  with `pt.Artifact.from_hf(...)` (see the `pyterrier-huggingface` skill).
- Close a cache (`cache.close()`) before copying its folder.
