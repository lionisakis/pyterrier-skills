# PISA with pyterrier-pisa

Checked with pyterrier-pisa 0.4.7 and PyTerrier 1.1.2.

## Contents
1. Install
2. Building an index
3. Retrieval
4. Learned sparse indexes
5. Errors

## 1. Install

```bash
pip install pyterrier-pisa
```

PyPI ships wheels for Linux x86_64 only. On macOS, Windows or ARM, use Terrier instead. PISA doesn't
need Java.

## 2. Building an index

```python
from pyterrier_pisa import PisaIndex

index = PisaIndex("collection.pisa", text_field="text", threads=8)
index.index(dataset.get_corpus_iter())
```

`PisaIndex(path, text_field=None, stemmer=None, index_encoding=None, batch_size=100000, stops=None,
threads=1, overwrite=False)`:

- `text_field`: the field to index. If it's left out, every `str` field is indexed and a warning is
  printed. Set it.
- `stemmer`: `porter2` for a new index (Terrier's default is `porter`, a different stemmer). Use
  `"none"` for learned sparse or non-English text.
- `stops`: the Terrier English stopword list by default. Pass `"none"` or a list to change it.
- `overwrite=True`, or `index.indexer(mode="overwrite")`, replaces an existing index.
- PISA stores **no document text**: `PisaIndex` has no `text_loader()`. Load text from the dataset
  (`dataset.text_loader()`) before a reranker.
- Prebuilt PISA indexes for MS MARCO and BEIR are on Hugging Face (see the `pyterrier-huggingface`
  skill). Load them with `PisaIndex.from_hf(...)` or `pt.Artifact.from_hf(...)`.

## 3. Retrieval

| Method | Defaults |
|---|---|
| `index.bm25(k1=0.9, b=0.4, num_results=1000)` | k1 0.9, b 0.4 (**not** Terrier's 1.2 / 0.75) |
| `index.dph(num_results=1000)` | |
| `index.pl2(c=1.0)` | |
| `index.qld(mu=1000.0)` | query likelihood, Dirichlet smoothing |
| `index.quantized()` | for learned sparse indexes (section 4) |

- Each scoring setting builds its own compressed structures on first use, e.g.
  `bm25.k1-0.9.b-0.4.q0.bmw.64.block_simdbp` inside the index folder. The first query with a new
  setting is slower, and every setting you try takes disk space.
- PISA results have no `docid` column, only `docno`.
- With default settings on the same corpus, PISA BM25 and Terrier BM25 give different scores, because
  their stemmer and k1/b both differ. On vaswani, nDCG@10 was 0.467 for PISA and 0.447 for Terrier. Pass
  matching parameters when you compare backends.

## 4. Learned sparse indexes

Index pre-weighted terms (e.g. from SPLADE) with `toks_indexer`, and query them with `quantized()`:

```python
index = PisaIndex("splade.pisa", stemmer="none")
(splade.doc_encoder() >> index.toks_indexer()).index(dataset.get_corpus_iter())
retriever = splade.query_encoder() >> index.quantized()
```

- `toks_indexer(scale=100.0)` multiplies float weights by `scale` and rounds them to integers.
- Use `stemmer="none"`. SPLADE terms are wordpieces and must not be stemmed.
- For the full SPLADE setup, read [splade.md](splade.md).

## 5. Errors

| Message | Cause | Fix |
|---|---|---|
| `RuntimeError: A PISA index already exists at … set mode="overwrite"` | indexing into an existing index | new path, `PisaIndex(path, overwrite=True)`, or `indexer(mode="overwrite")` |
| `UserWarning: text_field not specified; indexing all str fields` | no `text_field` | `PisaIndex(path, text_field="text")` |
| `pip install` finds no matching distribution | not Linux x86_64 | use Terrier |
| `AttributeError: … 'text_loader'` | PISA stores no text | use `dataset.text_loader()` |
