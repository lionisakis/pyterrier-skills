# Dense retrieval with pyterrier-dr

Checked against the source of pyterrier-dr 0.8.0 and pyterrier-alpha 0.16.5, with PyTerrier 1.1.2.

## Contents
1. Install
2. Encoders
3. Indexing
4. Choosing a retriever
5. Fusion with sparse retrieval
6. Problems

## 1. Install

```bash
pip install pyterrier-dr
pip install faiss-cpu     # only for the faiss_* retrievers (or faiss-gpu)
```

A GPU makes encoding orders of magnitude faster. On CPU, encoding even a few thousand documents
takes minutes, so use a prebuilt index (see the `pyterrier-huggingface` skill) where you can.

## 2. Encoders

Each encoder class has named variants that load a Hugging Face checkpoint:

| Class and variant | Checkpoint |
|---|---|
| `pyterrier_dr.TasB.dot()` | `sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco` |
| `pyterrier_dr.RetroMAE.msmarco_distill()` | `Shitao/RetroMAE_MSMARCO_distill` (used for the `pyterrier/*.retromae.flex` BEIR indexes; check the card) |
| `pyterrier_dr.RetroMAE.msmarco_finetune()` | `Shitao/RetroMAE_MSMARCO_finetune` |
| `pyterrier_dr.E5.base()`, `.small()`, `.large()` | `intfloat/e5-{base,small,large}-v2` |
| `pyterrier_dr.GTR.base()` (and `large`, `xl`, `xxl`) | `sentence-transformers/gtr-t5-*` |
| `pyterrier_dr.TctColBert(...)`, `Ance`, `BGEM3`, `Dragon`, `CDE`, ... | see the class docstrings |
| `pyterrier_dr.SBertBiEncoder("<name>")`, `HgfBiEncoder` | any sentence-transformers or Hugging Face model |

- An encoder is a transformer. `model.query_encoder()` (adds `query_vec`) and `model.doc_encoder()`
  (adds `doc_vec`) give the two halves. The model itself picks its role from its input: query encoder
  for topics, document encoder for documents, and **scorer** for results that have `query` and `text`.
  So `bm25 % 100 >> dataset.text_loader() >> model` reranks BM25 with the dense model, with no index.
- Use the same model, with the same variant, for documents and queries.

## 3. Indexing

```python
import pyterrier_dr

model = pyterrier_dr.TasB.dot()
index = pyterrier_dr.FlexIndex("collection.flex")
(model >> index).index(dataset.get_corpus_iter())
```

- `FlexIndex(path, sim_fn="dot")` stores the vectors and docnos. It does **not** store text or the
  model name, so record which model you used, e.g. in the path.
- Use `sim_fn="cos"` if the model was trained for cosine similarity. Check the model card.

## 4. Choosing a retriever

All of these are methods of `FlexIndex`, and each goes after the query encoder:
`model >> index.<method>()`.

| Method | Search | Use when |
|---|---|---|
| `retriever()` (= `np_retriever()`) | exact, numpy, CPU | up to a few million vectors, or as the reference |
| `torch_retriever()` | exact, torch | a GPU is available |
| `faiss_flat_retriever()` | exact, FAISS | FAISS is already in use |
| `faiss_hnsw_retriever(neighbours=32, ef_search=16)` | approximate, HNSW graph | large corpora with low latency. Builds a graph on first use and caches it on disk |
| `faiss_ivf_retriever(n_probe=1)` | approximate, inverted lists | very large corpora. Raise `n_probe` for recall |
| `voyager_retriever()`, `flatnav_retriever()`, `scann_retriever()` | approximate, other libraries | each needs its own package |
| `gar(...)`, `ladr(...)`, `pre_ladr(...)`, `ada_ladr(...)` | adaptive, corpus-graph based | reranking pipelines that expand candidates with a corpus graph |

- Approximate retrievers trade recall for speed. Compare their recall with `retriever()` on your topics
  before reporting results.
- Every retriever takes `num_results=1000`.

## 5. Fusion with sparse retrieval

```python
import pyterrier_alpha as pta

hybrid = pta.RRFusion(bm25, model >> index.retriever(), k=60, num_results=1000)

normalised_sum = (
    (bm25 >> pta.fusion.PerQueryMaxMinScore())
    + (model >> index.retriever() >> pta.fusion.PerQueryMaxMinScore())
)
```

- `RRFusion` scores each document by summing `1 / (rank + k)` over the inputs. Score scales don't
  matter.
- `PerQueryMaxMinScore` rescales each query's scores to 0–1 before they are added. Tune any weights on
  training topics.

## 6. Problems

- The pyterrier-dr README uses `pt.TerrierRetrieve` (legacy). Use `index.bm25()` or `pt.terrier.Retriever`.
- A how-to page uses `docid` where the column is `docno`.
- Indexing into an existing `FlexIndex` raises `RuntimeError: Index already exists at …`. Use a new path, or
  `(model >> index.indexer(mode="overwrite")).index(...)`.
