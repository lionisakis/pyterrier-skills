---
name: pyterrier-dense
description: >-
  Uses dense bi-encoder models in PyTerrier with pyterrier-dr, in every role:
  encoders (TAS-B, E5, RetroMAE, GTR, TCT-ColBERT, BGE-M3, any
  sentence-transformers model), indexing vectors in a FlexIndex, exact and
  approximate retrieval (numpy, torch, FAISS flat/HNSW/IVF), reranking any
  first stage with the same model as a scorer, and vector pseudo-relevance
  feedback (VectorPrf, AveragePrf). Covers matching query and document
  encoders, similarity functions and prebuilt .flex indexes. Use whenever
  dense retrieval, bi-encoders, embeddings, FlexIndex or pyterrier_dr appears
  in a PyTerrier task, whether as retriever, reranker or PRF. Not for
  monoELECTRA or other cross-encoders (see pyterrier-reranking), ColBERT
  (see pyterrier-colbert), or fusing dense with sparse retrieval (see
  pyterrier-fusion).
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-dr 0.8.0, pyterrier-alpha 0.16.5, CPU only)
---

# Dense retrieval (pyterrier-dr)

Assumes the `pyterrier-basics` skill.

Docs: [pyterrier-dr](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-dr/index.html) ·
[Artifacts](https://pyterrier.readthedocs.io/en/latest/artifacts/index.html)

## Choosing a dense role

| Role | Use | Needs |
|---|---|---|
| Retrieve | `model >> flex_index.retriever()` (or an approximate retriever) | a FlexIndex built with the same model |
| Rerank another first stage | `model` (or `model.scorer()`) after a text loader | only the checkpoint and document text |
| PRF on the query vector | `index.vec_loader() >> pyterrier_dr.VectorPrf(...) >> index.retriever()` | a FlexIndex |

Check for a prebuilt `.flex` index first (see the `pyterrier-huggingface` skill). Its card names the
encoder that built it.

## Install and encoder

```bash
pip install pyterrier-dr
pip install faiss-cpu     # only for the faiss_* retrievers (or faiss-gpu)
```

```python run slow
import itertools
import pyterrier as pt
import pyterrier_dr

dataset = pt.get_dataset("irds:vaswani")
docs = list(itertools.islice(dataset.get_corpus_iter(), 50))   # small slice: CPU encoding is slow
model = pyterrier_dr.TasB.dot()                                 # the encoder
```

The model picks its role from its input: query encoder for topics, document encoder for documents, and
scorer for results that have `query` and `text`. Any sentence-transformers checkpoint loads with
`pyterrier_dr.SBertBiEncoder("sentence-transformers/all-MiniLM-L6-v2")`. For the other encoder classes
and their checkpoints, read [references/dense.md](references/dense.md).

## Index and retrieve

```python run slow
dense_index = pyterrier_dr.FlexIndex("vaswani.flex")
if not dense_index.built():
    (model >> dense_index).index(docs)
dense = model >> dense_index.retriever()          # same encoder for queries
assert len(dense.search("chemical reactions")) > 0
```

- `FlexIndex` stores vectors and docnos only: no text, and no model name. Record the model, e.g. in the path.
- `retriever()` is exact search. For large corpora, choose an approximate retriever
  (`faiss_hnsw_retriever()`, `faiss_ivf_retriever()`, ...) and compare its recall with `retriever()`:
  read [references/dense.md](references/dense.md).

## Rerank

```python run slow
bm25 = pt.terrier.TerrierIndex.example().bm25(k1=1.2, b=0.75)
reranked = (bm25 % 10 >> dataset.text_loader() >> model.scorer()).search("chemical reactions")
assert len(reranked) == 10
```

- `model.scorer()` (alias `text_scorer()`) is the explicit form: it raises `InputValidationError` when
  `text` is missing, whereas `model` alone silently acts as a query encoder and passes the ranking through
  unchanged. It scores by dot product only (`sim_fn="cos"` raises `ValueError`).
- Bi-encoders are usually weaker rerankers than cross-encoders, but cheaper. For choosing the depth, see
  the `pyterrier-reranking` skill; for evaluating, the `pyterrier-experiments` skill.
- The `bm25` above is the prebuilt full-vaswani `TerrierIndex.example()`, a stand-in so the code runs. For a
  BM25 first stage over your own collection (or a slice of one), see the `pyterrier-retrieval` skill.

## Pseudo-relevance feedback

```python run slow
prf = (
    model >> dense_index.retriever()
    >> dense_index.vec_loader()                          # adds doc_vec to each result
    >> pyterrier_dr.VectorPrf(alpha=1, beta=0.2, k=3)    # new query_vec from the top k
    >> dense_index.retriever()
)
assert len(prf.search("chemical reactions")) > 0
```

`VectorPrf` adds `beta` × the mean of the top-`k` document vectors to `alpha` × the query vector
(Rocchio-style). `AveragePrf(k=3)` averages the query and top-`k` vectors. Both need the `query_vec`
column, which retrievers keep unless `drop_query_vec=True`.

## Don't

- Don't query a FlexIndex with a different model, or a different variant, from the one that built it.
- Don't assume `sim_fn`. Pass `FlexIndex(path, sim_fn="cos")` on **every** load if the card says cosine and
  the model doesn't normalise its vectors: `sim_fn` isn't stored in the index, and the `faiss_*` retrievers
  and `scorer()` always use dot product. The card may not say: check `np.linalg.norm` of one encoded batch.
  `SBertBiEncoder` itself encodes with `normalize_embeddings=False`; a model comes out unit-norm only if its
  `modules.json` has a `Normalize` module (all-MiniLM-L6-v2 and the bge-* models do) or it is an `E5` variant.
  On a unit-norm model, dot product equals cosine, so the dot-only `scorer()` ranks as `sim_fn="cos"` would.
- Don't trust `built()` after an interrupted indexing run. It only checks that the folder exists; delete
  the folder or use `indexer(mode="overwrite")`.
- Don't report approximate-retriever results without checking their recall against `retriever()`.
- Don't expect text from a FlexIndex. Use `dataset.text_loader()` before a reranker.
- Don't encode a large corpus on CPU. Use a GPU or a prebuilt index.
- Don't copy `pt.TerrierRetrieve` or `docid` from the pyterrier-dr README and how-to. Use `index.bm25()`
  and `docno`.
- Don't pass `drop_query_vec=True` to the first retriever of a PRF pipeline.

## Reference files

| Read | When |
|---|---|
| [references/dense.md](references/dense.md) | choosing an encoder or checkpoint, FlexIndex options, exact vs approximate retrievers, PRF parameters |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dense results look random | query encoder differs from the document encoder | use the index's encoder |
| `RuntimeError: Index already exists at <path>.flex` | indexing into an existing FlexIndex | check `built()`, new path, or `index.indexer(mode="overwrite")` |
| Reranked scores identical to the first stage, no error | `model` received results without `text` and acted as a query encoder (FlexIndex stores no text) | add `dataset.text_loader()`; use `model.scorer()` to get an `InputValidationError` instead |
| `Exception: Problem applying … VectorPrf.transform … for qid …` (cause: `InputValidationError` naming `query_vec`) | the first retriever dropped `query_vec` | remove `drop_query_vec=True` |
| `AssertionError: faiss required; install using instructions here …` | faiss is not a dependency | `pip install faiss-cpu` (or `faiss-gpu`) |
| Indexing takes hours | CPU encoding | GPU, a smaller corpus, or a prebuilt index |
