# Late interaction with pyterrier-pylate (PyLate)

Checked with https://github.com/lightonai/pyterrier-pylate at commit `1eb3085` (2026-07-13, version
0.1.0), pylate 1.6.0, fast-plaid 1.4.6, transformers 5.3, sentence-transformers 5.3, torch 2.11 (CPU)
and PyTerrier 1.1.2, **on CPU only**. Indexing the first 300 vaswani documents, end-to-end retrieval,
the `% k` cutoff, both scorers after BM25, `pt.Experiment`, reloading with `pt.Artifact.load`, the
"already exists" error and `mode="overwrite"` were all run. `to_hf` was not run. Indexing took about
6 s per document on a sandbox limited to 0.2 CPU; a normal CPU is far faster, and a GPU faster still.

## Contents
1. Install
2. Transformers and columns
3. Building an index
4. Retrieval
5. Reranking: index scorer and text scorer
6. Sharing and loading
7. Known problems

## 1. Install

```bash
pip install 'pyterrier[java]' ir-datasets git+https://github.com/lightonai/pyterrier-pylate.git
```

- `ir-datasets` is needed for `pt.get_dataset("irds:...")` below; neither the git install nor
  `pyterrier[java]` brings it in.

- The README says `pip install pyterrier-pylate`, but the package is not on PyPI (checked
  2026-10-08). Install from git.
- The git install depends on plain `pyterrier>=1.0`, without the Java extra. Add `pyterrier[java]` (or
  `pyterrier[all]`), or every Terrier call, such as the BM25 in section 5, fails with
  `ImportError: pyterrier[java] not installed`.
- It pulls in `pylate>=1.3`, which pins `transformers<=5.3.0,>=4.41.0` (installs 5.3) and
  `sentence-transformers==5.3.0`. It therefore conflicts with `pyterrier_colbert2` (`transformers<5`) and
  with anything needing a newer transformers 5.x, so the two backends can't share an environment. Make the
  environment fully isolated: a venv that reaches another venv through `.pth` or `--system-site-packages`
  fails on import with `RuntimeError: operator torchvision::nms does not exist`.
- No compiler, FAISS or GPU is needed. PLAID search uses `fast-plaid` (prebuilt wheels).
- `import pyterrier_pylate` is cheap; PyLate is imported when a model or index is first used.

## 2. Transformers and columns

| Transformer | Input | Output |
|---|---|---|
| `model.query_encoder()` | `qid, query` | `+ query_embs` |
| `model.doc_encoder()` | `docno, text` | `+ doc_embs` |
| `model.text_scorer()` | `qid, query` or `query_embs`, `docno`, `text` or `doc_embs` | `+ score, rank` |
| `index.retriever(k=1000)` | `qid, query_embs` | `qid, docno, score, rank` |
| `index.scorer()` | `qid, query_embs, docno` | `+ score, rank` |

`query_embs` and `doc_embs` hold one `(num_tokens, dim)` NumPy matrix per row. `model = PyLateBiEncoder(name,
batch_size=32, text_field="text", verbose=False, device=None, **colbert_args)`; extra keyword arguments go
to `pylate.models.ColBERT`.

## 3. Building an index

```python
import pyterrier as pt
from pyterrier_pylate import PyLateBiEncoder, PlaidIndex

dataset = pt.get_dataset("irds:vaswani")
model = PyLateBiEncoder("lightonai/GTE-ModernColBERT-v1")      # device="cpu" to force CPU
index = PlaidIndex("./vaswani.plaid")
(model.doc_encoder() >> index).index(dataset.get_corpus_iter())
```

- The index stores token embeddings only, not text, and not the model name. Record the checkpoint
  yourself (on the Hub card, for example); queries must use the same model.
- An existing non-empty folder raises `RuntimeError: Index already exists`. Rebuild with
  `(model.doc_encoder() >> index.indexer(mode="overwrite")).index(...)`.
- `index.indexer(batch_size=100_000)` is how many documents are buffered per `add_documents` call.
- `PlaidIndex(path, nbits=..., device=..., ...)` forwards keyword arguments to `pylate.indexes.PLAID`
  (in pylate 1.6: `nbits=4`, `kmeans_niters=4`, `n_ivf_probe=8`, `n_full_scores=8192`, `use_fast=True`).
  Verify the defaults against the installed `pylate/indexes/plaid.py`. The Stanford PLAID knobs
  (`ncells`, `centroid_score_threshold`, `ndocs`) are ignored by the default fast backend.
- Search settings are constructor arguments, so pass the same ones when reopening an index to compare
  runs.

## 4. Retrieval

```python
retriever = model.query_encoder() >> index.retriever(k=1000)
results = retriever.search("chemical reactions")
results = retriever.transform(dataset.get_topics())    # several queries: a topics frame, not a .search() loop
```

- The retriever needs `query_embs`, so always put the query encoder in front of it.
- Ranks start at 0 (as for most PyTerrier retrievers).
- PLAID search is approximate: it can return fewer rows than the index holds (59 of 60 here). If the count
  matters, widen the search settings passed to `PlaidIndex(...)` (section 3).
- `retriever % 10` is rewritten into `index.retriever(k=10)`.

## 5. Reranking: index scorer and text scorer

```python
bm25 = pt.Artifact.from_hf("pyterrier/vaswani.terrier").bm25()

# documents already in a PlaidIndex: no document re-encoding
from_index = bm25 % 100 >> model.query_encoder() >> index.scorer()

# no index at all: encodes each candidate's text at query time
from_text = bm25 % 100 >> dataset.text_loader() >> model.text_scorer()
```

- `index.scorer()` drops candidates whose `docno` is not in the index, so it can return fewer rows than
  it received. Use it only when the index covers the first stage's collection.
- `text_scorer()` re-encodes every candidate document for every query; cost grows with the rerank depth.
- Both need a `qid` column to group candidates by query.

## 6. Sharing and loading

`PlaidIndex` is an artifact (type tag `multivec_index.plaid`), registered by the package's entry point:

```python
index.to_hf("owner/collection.model.plaid")
index = pt.Artifact.from_hf("owner/collection.model.plaid")      # or PlaidIndex.from_hf(...)
index = pt.Artifact.load("./vaswani.plaid")                       # a folder on disk
```

For the Hub workflow, see the `pyterrier-huggingface` skill.

## 7. Known problems

- Not on PyPI, although the README says `pip install pyterrier-pylate`. Install from git.
- `index.scorer()` silently drops candidates whose `docno` is not in the index. Reranking BM25 top-20 for
  3 vaswani topics against a 300-document index kept 1 row of 60. Check `len` before and after, or use
  `model.text_scorer()`, which scored all 60.
- Results of `model.query_encoder() >> index.scorer()` (or `index.retriever()`) keep the `query_embs` column
  (one matrix per row). It is large; drop it with
  `pt.apply.generic(lambda df: df.drop(columns=["query_embs"], errors="ignore"))` before saving runs or
  caching results. The `text_scorer()` path has no such column, hence `errors="ignore"`.
- The index stores no model name (`pt_meta.json` holds only type, format, package hint and `doc_count`).
- `index.indexer(mode="overwrite")` deletes the whole folder before indexing.
- The package uses `pyterrier_alpha` (`pta.validate`, `pta.Artifact`), which it installs itself.
- On a machine whose NVIDIA driver is older than the torch wheel, every model load prints `UserWarning: CUDA
  initialization: The NVIDIA driver on your system is too old`, even with `device="cpu"`. It is harmless on
  CPU; the `pyterrier-basics` Troubleshooting table says how to silence it or use the GPUs.
