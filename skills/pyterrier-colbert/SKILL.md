---
name: pyterrier-colbert
description: >-
  Uses ColBERT v2 late-interaction models in PyTerrier with pyterrier_colbert2,
  in every role: loading a prebuilt PLAID index from Hugging Face, building a
  ColBERT v2 index, end-to-end retrieval with PLAID settings (ncells,
  centroid_score_threshold, ndocs), reranking any first stage with the
  index-free text_scorer, and ColBERT pseudo-relevance feedback (PLAID-PRF)
  with its weighting schemes. Covers install on GPU and CPU, the transformers<5
  cap and known bugs. Use whenever ColBERT, ColBERTv2, PLAID, late interaction
  or ColBERT PRF appears in a PyTerrier task, whether as retriever, reranker or
  expansion. Not for choosing between model families or fusing ColBERT with
  other retrievers (see pyterrier-pipelines).
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier_colbert2 4947df6, colbert-ai 0.2.22, transformers 4.57, CPU only)
---

# ColBERT v2 / PLAID

Assumes the `pyterrier-basics` skill.

Docs: [pyterrier_colbert2](https://github.com/cmacdonald/pyterrier_colbert2) ·
[ColBERT](https://github.com/stanford-futuredata/ColBERT) ·
[Artifacts](https://pyterrier.readthedocs.io/en/latest/artifacts/index.html)

## Choosing a ColBERT role

| Role | Use | Needs | Read |
|---|---|---|---|
| Retrieve | `index.end_to_end(k=...)` | a ColBERT v2 index (prebuilt or built) | Retrieve, below |
| Rerank another first stage | `ColBERTModelOnlyFactory(...).text_scorer()` | only the checkpoint and document text | Rerank, below |
| PRF on ColBERT embeddings | `index.plaid_prf_end_to_end(...)` | a ColBERT v2 index in a writable folder | [references/plaid-prf.md](references/plaid-prf.md) |

The reranker needs no index, so it is the cheapest way to try ColBERT on a new collection.

## Install

```bash
pip install git+https://github.com/cmacdonald/pyterrier_colbert2.git
conda install -c pytorch -c conda-forge faiss-cpu    # or faiss-gpu; not installed automatically
```

- A GPU is recommended. Everything also runs on CPU, very slowly, and needs `g++` and `ninja` to
  compile ColBERT's extensions on first use.
- It caps `transformers<5`, which conflicts with recent pyterrier-dr. Use a separate environment if
  pip can't resolve both.
- It replaces the old `pyterrier_colbert`: `ColBERTFactory` and `ColBERTIndexer` don't exist here.

For the full list of install caveats, read [references/colbert2.md](references/colbert2.md).

## Index

Check for a prebuilt index first (see the `pyterrier-huggingface` skill):

```python
import pyterrier as pt
import pyterrier_colbert2   # registers the dense_index.colbert artifact type

index = pt.Artifact.from_hf(
    "pyterrier/msmarco_psg_v1.colbertv2",    # 24 GB download
    plaid_mode=True, ncells=4, centroid_score_threshold=0.4, ndocs=4096,
)
```

The PLAID settings are illustrative. Take them from the task or paper, and pass them explicitly.

To build an index instead:

```python
from pyterrier_colbert2.indexing import ColbertV2Indexer

indexer = ColbertV2Indexer(
    index_location="/path/to/indexes", checkpoint="colbert-ir/colbertv2.0",
    index_name="my_index", nbits=2,
)
index = indexer.index(dataset.get_corpus_iter())   # loads the whole corpus into memory
```

Very small collections fail; about 200 documents was the smallest that worked.

## Retrieve

```python
retriever = index.end_to_end(k=1000)
results = retriever.search("what are chemical reactions?")
```

- `plaid_mode=False` is not brute force. It uses the same PLAID engine with ColBERT's default settings.
- `end_to_end(decompose=True)` (PLAID mode only) returns the four PLAID stages as separate transformers.
- Queries are processed one at a time.

## Rerank

```python
from pyterrier_colbert2.ranking import ColBERTModelOnlyFactory

factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0")        # gpu=False for CPU
pipeline = bm25 % 100 >> dataset.text_loader() >> factory.text_scorer()
```

The scorer needs `query` and `text` columns. The rerank depth (`% 100`) comes from the task; for how to
choose it and evaluate the reranker, see the `pyterrier-pipelines` skill.

## Pseudo-relevance feedback (PLAID-PRF)

```python
prf = index.plaid_prf_end_to_end(k=1000, top_psg=3, top_exp=14, beta=0.7)
```

`plaid_prf_end_to_end(k, **kwargs)` is `plaid_prf(index, **kwargs) >> index.end_to_end(k=k, query_encoded=True)`:
it adds expansion embeddings from the top passages to the query, then retrieves again. Compare it against
`index.end_to_end()` with the same PLAID settings. For every parameter and weighting scheme, read
[references/plaid-prf.md](references/plaid-prf.md).

## Don't

- Don't import `ColBERTFactory` or `ColBERTIndexer`. They belong to the old `pyterrier_colbert`.
- Don't install pyterrier_colbert2 next to packages that need transformers 5. Use a separate environment.
- Don't chain `index.query_encoder()` into `end_to_end(query_encoded=True)`. The first writes `query_embs`,
  the second reads `query_vec`.
- Don't use `weighting="dfr_rsj"` in PLAID-PRF. Use `"dfr"`.
- Don't run PLAID-PRF on a read-only index folder. The first call writes statistics files into it.
- Don't compare PLAID runs with different `ncells`, `centroid_score_threshold` or `ndocs` as if they
  were the same system.
- Don't trust `rank` across ColBERT methods. Some start at 0, some at 1; recompute with `pt.model.add_ranks`.

## Reference files

| Read | When |
|---|---|
| [references/colbert2.md](references/colbert2.md) | installing, loading, building or querying a ColBERT v2 index, or using the text scorer |
| [references/plaid-prf.md](references/plaid-prf.md) | configuring PLAID-PRF: feedback depth, expansion size, weighting |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError` on `import pyterrier_colbert2` | ColBERT or torch missing; the package imports them at once | install `colbert-ai` (pulled in by the git install) and torch |
| pip can't resolve transformers | `pyterrier_colbert2` caps `transformers<5` | separate environment |
| C++ compile errors on first use | no `g++` or `ninja` on a CPU machine | install them, or use a GPU |
| Indexing fails on a tiny collection | too few documents for ColBERT's clustering | index at least about 200 documents |
| `Exception: Problem applying … for qid` in PLAID-PRF | `weighting="dfr_rsj"` | `weighting="dfr"` |
| First PLAID-PRF call is very slow or fails with a permission error | it scans the index and writes `stats.json` and friends | writable index folder; run once to warm up |
| `text_scorer(query_encoded=True)` floods the log | debugging prints in the package | ignore, or use `query_encoded=False` |
