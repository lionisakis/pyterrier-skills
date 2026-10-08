# ColBERT v2 / PLAID with pyterrier_colbert2

Read from the source of https://github.com/cmacdonald/pyterrier_colbert2 at commit `4947df6`
(2026-09-21). It needs a GPU, so this page was **not** run during writing. Check every call against
the README of the version you install.

## Contents
1. Install
2. Loading a prebuilt index
3. Retrieval
4. Building an index
5. Reranking without an index
6. Known problems

## 1. Install

```bash
pip install git+https://github.com/cmacdonald/pyterrier_colbert2.git
conda install -c pytorch -c conda-forge faiss-cpu    # or faiss-gpu; not installed automatically
```

- It needs a CUDA GPU for inference, Linux (because of FAISS) and Python 3.9 or newer. The whole index
  must fit in memory: MS MARCO passage takes about 28 GB.
- It installs `colbert-ai` from ColBERT's git master, unpinned, and caps `transformers<5`. That cap
  conflicts with packages that need transformers 5 (such as recent pyterrier-dr). Use a separate
  environment if pip can't resolve both.
- `import pyterrier_colbert2` imports ColBERT and torch straight away, so ColBERT must be installed
  even to import the package.
- It replaces the old `pyterrier_colbert`: `ColBERTFactory` and `ColBERTIndexer` don't exist here.

## 2. Loading a prebuilt index

```python
import pyterrier as pt
import pyterrier_colbert2   # registers the dense_index.colbert artifact type

index = pt.Artifact.from_hf(
    "pyterrier/msmarco_psg_v1.colbertv2",    # 24 GB download
    plaid_mode=True, ncells=4, centroid_score_threshold=0.4, ndocs=4096,
)
```

Keyword arguments go to `ColBERTv2Index(index_location, plaid_mode=False, colbert=None, ncells=None,
centroid_score_threshold=None, ndocs=None, gpu=True, ...)`. The checkpoint is read from the index
metadata unless you pass `colbert=`.

## 3. Retrieval

```python
retriever = index.end_to_end(k=1000)
results = retriever.search("what are chemical reactions?")
```

- `end_to_end(k=1000, decompose=False, query_encoded=False)`.
- `decompose=True` (PLAID mode only) returns the four PLAID stages as separate transformers, for
  efficiency experiments.
- Queries are processed one at a time.
- `plaid_mode=False` is not brute force. It uses the same PLAID engine with ColBERT's default settings.

## 4. Building an index

```python
from pyterrier_colbert2.indexing import ColbertV2Indexer

indexer = ColbertV2Indexer(
    index_location="/path/to/indexes",
    checkpoint="colbert-ir/colbertv2.0",
    index_name="my_index",
    nbits=2,
)
index = indexer.index(dataset.get_corpus_iter())   # returns a ColBERTv2Index (plaid_mode off)
```

- The whole corpus is loaded into memory before indexing.
- Very small collections fail. The tests needed at least about 200 documents.
- Publish it with `index.to_hf("owner/name")` (see the `pyterrier-huggingface` skill).

## 5. Reranking without an index

```python
from pyterrier_colbert2.ranking import ColBERTModelOnlyFactory

factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0")
pipeline = bm25 % 100 >> dataset.text_loader() >> factory.text_scorer()
```

This is a reranker, so see the `pyterrier-reranking` skill for how to build the pipeline around it.
`gpu=False` runs on CPU, slowly.

## 6. Known problems

- `query_encoder()` writes a `query_embs` column, but `end_to_end(query_encoded=True)` reads
  `query_vec`, so the two can't be chained. Only PLAID-PRF produces `query_vec` (see the
  `pyterrier-expansion` skill).
- Rank numbering differs between methods (some start at 0, some at 1). Recompute ranks with
  `pt.model.add_ranks` if a later stage depends on them.
- `text_scorer(query_encoded=True)` prints debugging lines for every query.
