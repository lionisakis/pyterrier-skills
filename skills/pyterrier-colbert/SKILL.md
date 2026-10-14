---
name: pyterrier-colbert
description: >-
  Uses ColBERT v2 late-interaction models in PyTerrier with pyterrier_colbert2,
  in every role: loading a prebuilt PLAID index from Hugging Face, building a
  ColBERT v2 index, end-to-end retrieval with PLAID settings (ncells,
  centroid_score_threshold, ndocs), reranking any first stage with the
  index-free text_scorer, and ColBERT pseudo-relevance feedback (PLAID-PRF)
  with its weighting schemes. Also covers pyterrier-pylate, the PyLate backend
  for multi-vector models such as GTE-ModernColBERT: encoders, a PlaidIndex
  artifact, retrieval, and index or text scorers. Covers install on GPU and CPU,
  the clashing transformers requirements and known bugs. Use whenever ColBERT,
  ColBERTv2, PLAID, PyLate, multi-vector, late interaction or ColBERT PRF
  appears in a PyTerrier task, whether as retriever, reranker or expansion. Not
  for choosing between model families (see pyterrier-retrieval) or fusing ColBERT
  with other retrievers (see pyterrier-fusion).
compatibility: pyterrier>=1.1 (checked with 1.1.2; pyterrier_colbert2 4947df6, colbert-ai 0.2.22, transformers 4.57; pyterrier-pylate 1eb3085, pylate 1.6.0, transformers 5.3; CPU only)
---

# Late interaction: ColBERT v2 / PLAID and PyLate

Assumes the `pyterrier-basics` skill.

Docs: [pyterrier_colbert2](https://github.com/cmacdonald/pyterrier_colbert2) ·
[pyterrier-pylate](https://github.com/lightonai/pyterrier-pylate) ·
[PyLate](https://lightonai.github.io/pylate/) ·
[ColBERT](https://github.com/stanford-futuredata/ColBERT) ·
[Artifacts](https://pyterrier.readthedocs.io/en/latest/artifacts/index.html)

## Choosing a backend

| | `pyterrier_colbert2` | `pyterrier-pylate` |
|---|---|---|
| Models | Stanford ColBERT checkpoints (`colbert-ir/colbertv2.0`) | PyLate checkpoints (`lightonai/GTE-ModernColBERT-v1`, ...); PyLate's loader also converts Stanford checkpoints (not checked here) |
| transformers | `<5` | `<=5.3.0,>=4.41.0` (pinned by pylate 1.6; installs 5.3) |
| Prebuilt index on the Hub | `pyterrier/msmarco_psg_v1.colbertv2` and others | none (no `multivec_index.plaid` artifacts, checked 2026-10-08); build one |
| Retrieve, rerank | yes | yes, plus rerank from indexed embeddings |
| PLAID-PRF | yes | no |
| Install | git, plus FAISS; C++ compile on CPU | git; no FAISS or compiler |

The two need different transformers versions, so use a separate, **fully isolated** environment for each: no
`--system-site-packages` and no `.pth` file pointing at another venv. Indexes are not interchangeable between
them. Pick `pyterrier_colbert2` for a prebuilt `.colbertv2` index or PLAID-PRF, and `pyterrier-pylate` for
PyLate models (ModernColBERT and others). The rest of this page covers `pyterrier_colbert2` until the
"PyLate backend" section.

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

Very small collections fail; about 200 documents was the smallest that worked. (This is a
pyterrier_colbert2 limit; PyLate's PLAID index built from 60 documents.)

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
choose it, see the `pyterrier-reranking` skill, and for evaluating the reranker, the `pyterrier-experiments` skill.

## Pseudo-relevance feedback (PLAID-PRF)

```python
prf = index.plaid_prf_end_to_end(k=1000, top_psg=3, top_exp=14, beta=0.7)
```

`plaid_prf_end_to_end(k, **kwargs)` is `plaid_prf(index, **kwargs) >> index.end_to_end(k=k, query_encoded=True)`:
it adds expansion embeddings from the top passages to the query, then retrieves again. Compare it against
`index.end_to_end()` with the same PLAID settings. For every parameter and weighting scheme, read
[references/plaid-prf.md](references/plaid-prf.md).

## PyLate backend (pyterrier-pylate)

```bash
pip install 'pyterrier[java]' ir-datasets git+https://github.com/lightonai/pyterrier-pylate.git   # not on PyPI; the git install alone has no Java extra or ir-datasets
```

A `PyLateBiEncoder` is a factory for encoders and a scorer. A `PlaidIndex` stores token embeddings and
gives a retriever and a scorer. Both read and write `query_embs` and `doc_embs` columns.

```python
import pyterrier as pt
from pyterrier_pylate import PyLateBiEncoder, PlaidIndex

model = PyLateBiEncoder("lightonai/GTE-ModernColBERT-v1")
index = PlaidIndex("./collection.plaid")
(model.doc_encoder() >> index).index(dataset.get_corpus_iter())

retriever = model.query_encoder() >> index.retriever(k=1000)
from_index = bm25 % 100 >> model.query_encoder() >> index.scorer()                  # embeddings from the index
from_text = bm25 % 100 >> dataset.text_loader() >> model.text_scorer()             # no index needed
```

- Always put `model.query_encoder()` in front of `index.retriever()` or `index.scorer()`.
- The index doesn't record the model. Query with the checkpoint that built it, and write it on the Hub card.
- `index.scorer()` drops candidates that aren't in the index.
- PLAID search is approximate: `index.retriever()` can return fewer rows than the index holds (59 of 60).
- `PlaidIndex` is a `multivec_index.plaid` artifact: `index.to_hf(...)`, `pt.Artifact.from_hf(...)`.

For the transformer table, index options and every caveat, read [references/pylate.md](references/pylate.md).

## Don't

- Don't import `ColBERTFactory` or `ColBERTIndexer`. They belong to the old `pyterrier_colbert`.
- Don't install pyterrier_colbert2 next to pyterrier-pylate (transformers `<5` against `<=5.3`). Use a
  separate environment for each, fully isolated: no `--system-site-packages`, no `.pth` into another venv.
- Don't `pip install pyterrier-pylate` from PyPI; it isn't published. Install it from git.
- Don't load a `PlaidIndex` with `pyterrier_colbert2` or a `.colbertv2` index with pyterrier-pylate.
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
| [references/pylate.md](references/pylate.md) | anything with pyterrier-pylate: install, columns, index options, scorers, sharing |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError` on `import pyterrier_colbert2` | ColBERT or torch missing; the package imports them at once | install `colbert-ai` (pulled in by the git install) and torch |
| pip can't resolve transformers | `pyterrier_colbert2` caps `transformers<5`; pylate pins `<=5.3.0,>=4.41.0`, which also clashes with newer 5.x | separate environment per backend |
| `No matching distribution found for pyterrier-pylate` | not on PyPI | `pip install git+https://github.com/lightonai/pyterrier-pylate.git` |
| `ImportError: pyterrier[java] not installed … triggered by Retriever.__init__` on a Terrier call in the PyLate environment | the pylate git install pulls plain `pyterrier`, without the Java extra | `pip install 'pyterrier[java]'` (or `'pyterrier[all]'`) in that environment |
| `RuntimeError: operator torchvision::nms does not exist`, then `Could not import module 'PreTrainedModel'` on `import pyterrier_pylate` | a `torchvision` from another environment leaks in (`.pth` or `--system-site-packages`); its `RuntimeError` slips past transformers' `except ImportError` | rebuild a fully isolated environment |
| PyLate `index.retriever()` or `index.scorer()` fails validation on `query_embs` | no query encoder in front | `model.query_encoder() >> index.retriever(k=...)` |
| PyLate `index.scorer()` returns fewer rows than it received | candidates not in the `PlaidIndex` are dropped | index the whole collection the first stage searches, or use `model.text_scorer()` |
| `RuntimeError: Index already exists` (PyLate) | the `PlaidIndex` folder is not empty | `index.indexer(mode="overwrite")`, or a new folder |
| C++ compile errors on first use | no `g++` or `ninja` on a CPU machine | install them, or use a GPU |
| Indexing fails on a tiny collection (pyterrier_colbert2) | too few documents for ColBERT's clustering | index at least about 200 documents; PyLate has no such floor |
| `Exception: Problem applying … for qid` in PLAID-PRF | `weighting="dfr_rsj"` | `weighting="dfr"` |
| First PLAID-PRF call is very slow or fails with a permission error | it scans the index and writes `stats.json` and friends | writable index folder; run once to warm up |
| `text_scorer(query_encoded=True)` floods the log | debugging prints in the package | ignore, or use `query_encoded=False` |
