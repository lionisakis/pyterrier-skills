---
name: pyterrier-first-stage
description: >-
  Builds and queries first-stage retrievers in PyTerrier: Terrier indexes and
  BM25/DPH/PL2 retrieval, PISA indexes, learned sparse retrieval with SPLADE,
  dense retrieval with pyterrier-dr (TAS-B, E5, RetroMAE, TCT-ColBERT and
  FlexIndex with numpy, FAISS or HNSW retrievers), late interaction with ColBERT
  v2 / PLAID, and hybrid fusion of sparse and dense results. Covers indexing
  options, metadata lengths, stemming, retriever parameters and choosing a
  backend. Use when indexing a corpus, choosing or configuring a retriever, or
  combining retrievers in PyTerrier. Not for reranking, query expansion or
  evaluation.
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-pisa 0.4.7, pyterrier-dr 0.8.0, pyterrier-splade 0.1.0, pyterrier-alpha 0.16.5)
---

# First-stage retrieval

Assumes the `pyterrier-basics` skill.

Docs: [Terrier indexing](https://pyterrier.readthedocs.io/en/latest/terrier/indexing.html) ·
[Terrier retrieval](https://pyterrier.readthedocs.io/en/latest/terrier/retrieval.html) ·
[PISA](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier_pisa/index.html) ·
[SPLADE](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-splade/index.html) ·
[pyterrier-dr](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-dr/index.html) ·
[ColBERT v2](https://github.com/cmacdonald/pyterrier_colbert2)

## Choosing a first stage

Check for a prebuilt index first (see the `pyterrier-huggingface` skill). Build one only if none fits.

| Need | Default choice | Install | Read |
|---|---|---|---|
| Lexical (BM25) retrieval, any platform | Terrier | core, Java 11+ | [references/terrier.md](references/terrier.md) |
| Fast lexical retrieval on large corpora | PISA | `pyterrier-pisa` (Linux x86_64) | [references/pisa.md](references/pisa.md) |
| Learned sparse retrieval | SPLADE, stored in PISA or Terrier | `pyterrier-splade` | [references/splade.md](references/splade.md) |
| Dense (bi-encoder) retrieval | pyterrier-dr `FlexIndex` | `pyterrier-dr` | [references/dense.md](references/dense.md) |
| Late interaction | ColBERT v2 / PLAID | `pyterrier_colbert2` (GPU recommended; runs slowly on CPU) | [references/colbert2.md](references/colbert2.md) |
| Sparse and dense together | reciprocal rank fusion | `pyterrier-alpha` | Fusion, below |

Read the reference file for a backend before writing code for it.

## Quick start: Terrier

```python run
import pyterrier as pt

dataset = pt.get_dataset("irds:vaswani")
index = pt.terrier.TerrierIndex("vaswani.terrier")
if not index.built():
    index.indexer(meta={"docno": 20, "text": 4096}).index(dataset.get_corpus_iter())

bm25 = index.bm25(k1=1.2, b=0.75)               # parameters stated, not assumed
results = (bm25 % 10 >> index.text_loader()).search("chemical reactions")
assert "text" in results.columns
```

- `TerrierIndex(path).index(docs)` is fine for a quick test. It guesses metadata lengths from the
  **first document only** and silently truncates longer docnos and text. For a real collection, call
  `.indexer(meta={...})` with widths chosen from the corpus.
- Keep the text in `meta` if rerankers will need it. Otherwise load text from the dataset.
- Create the index object once and reuse it for every retriever.

## Quick start: PISA

```python run
from pyterrier_pisa import PisaIndex

pisa = PisaIndex("vaswani.pisa", text_field="text", threads=4)
if not pisa.built():
    pisa.index(dataset.get_corpus_iter())
pisa_bm25 = pisa.bm25(k1=1.2, b=0.75)            # PISA's own default is k1=0.9, b=0.4
assert len(pisa_bm25.search("chemical reactions")) > 0
```

PISA stores no text and has no `text_loader()`. Use `dataset.text_loader()` before a reranker.

## Dense retrieval (pyterrier-dr)

```python run slow
import itertools
import pyterrier_dr

model = pyterrier_dr.TasB.dot()                  # the encoder
dense_index = pyterrier_dr.FlexIndex("vaswani.flex")
docs = list(itertools.islice(dataset.get_corpus_iter(), 100))   # small slice: CPU encoding is slow
(model >> dense_index).index(docs)
dense = model >> dense_index.retriever()         # same encoder for queries
assert len(dense.search("chemical reactions")) > 0
```

- Queries **must** be encoded with the model that encoded the documents. A prebuilt `.flex` index names
  its model on its card.
- `dense_index.retriever()` is exact search. For large corpora, choose an approximate retriever
  (`faiss_hnsw_retriever()`, `faiss_ivf_retriever()`, ...): read
  [references/dense.md](references/dense.md).
- Encoding a corpus on CPU is very slow. Use a GPU or a prebuilt index for anything beyond a few
  thousand documents.

## Learned sparse retrieval (SPLADE)

```python
import pyterrier_splade
from pyterrier_pisa import PisaIndex

splade = pyterrier_splade.Splade()
sparse_index = PisaIndex("splade.pisa", stemmer="none")
(splade.doc_encoder() >> sparse_index.toks_indexer()).index(dataset.get_corpus_iter())
splade_retriever = splade.query_encoder() >> sparse_index.quantized()
```

Index with `stemmer="none"`, and use the same SPLADE model for queries and documents. For the Terrier
variant and the prebuilt SPLADE caches, read [references/splade.md](references/splade.md).

## Fusion

```python run slow
import pyterrier_alpha as pta

hybrid = pta.RRFusion(bm25, dense, k=60)         # reciprocal rank fusion
assert len(hybrid.search("chemical reactions")) > 0
```

- Reciprocal rank fusion uses ranks, so the two score scales don't matter. Prefer it to `bm25 + dense`.
- To add scores instead, normalise each side first:
  `pta.fusion.PerQueryMaxMinScore()` on each input, then `+` with weights tuned on training topics.

## Parameters

Never rely on default parameters. They differ between backends: BM25 defaults differ between Terrier
and PISA, as do their stemmers. Pass every parameter explicitly, and check the values actually used
against the installed source, the index metadata or the paper. For indicative values and how to check
them, read [references/defaults.md](references/defaults.md).

## Don't

- Don't build an index when a suitable prebuilt one exists.
- Don't create `pt.terrier.Retriever(index)` without `wmodel=`. The default is DPH.
- Don't index a real collection with `TerrierIndex(path).index(...)`'s guessed metadata lengths.
- Don't give `pt.IterDictIndexer` a relative path. Use `os.path.abspath(path)`, or `TerrierIndex`.
- Don't expect to add documents to an existing Terrier or PISA index. Rebuild it.
- Don't query a dense or SPLADE index with a different model from the one that built it.
- Don't stem SPLADE terms. Build the index with `stemmer="none"`.
- Don't add raw BM25 and dense scores. Use `RRFusion` or normalise first.
- Don't call `text_loader()` on a PISA index. It stores no text.
- Don't compare Terrier and PISA (or Anserini) BM25 without matching k1, b and stemming.

## Reference files

| Read | When |
|---|---|
| [references/terrier.md](references/terrier.md) | building or querying a Terrier index: metadata, tokenisation, stemming, fields, positions, errors |
| [references/pisa.md](references/pisa.md) | building or querying a PISA index, or storing learned sparse weights in PISA |
| [references/splade.md](references/splade.md) | SPLADE indexing and retrieval |
| [references/dense.md](references/dense.md) | dense encoders, FlexIndex, choosing exact or approximate search |
| [references/colbert2.md](references/colbert2.md) | ColBERT v2 / PLAID retrieval or indexing |
| [references/defaults.md](references/defaults.md) | choosing or reporting BM25 and other parameters |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: an index is already built at this path` | `TerrierIndex(path).index(...)` on an existing index | check `index.built()` first, or use a new path |
| `RuntimeError: A PISA index already exists at …` | indexing into an existing PISA index | `PisaIndex(path, overwrite=True)` or a new path |
| `RuntimeError: Index already exists at <path>.flex` | indexing into an existing FlexIndex | new path, or `index.indexer(mode="overwrite")` |
| `JavaException … Cannot create new index: path …/var/<name> does not exist` | `IterDictIndexer` given a relative path | `os.path.abspath(path)` |
| Java log `Term … written in N bytes. Max expected size was M`; docnos or text cut short | metadata wider than its `meta` width | rebuild with larger `meta` widths |
| BM25 scores differ between backends on the same data | different k1, b or stemmer | pass the same parameters; see references/defaults.md |
| `KeyError: 'text'` in the next stage | the index stores no text | `dataset.text_loader()` |
| Dense results look random | query encoder differs from the document encoder | use the index's encoder |
| `pip install pyterrier-pisa` finds no distribution | not Linux x86_64 | use Terrier |
| Dense indexing takes hours | CPU encoding | GPU, a smaller corpus, or a prebuilt index |
