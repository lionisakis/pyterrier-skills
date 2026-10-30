# Learned sparse retrieval with SPLADE

Checked with pyterrier-splade 0.1.0, pyterrier-pisa 0.4.7 and PyTerrier 1.1.2, on a small vaswani slice
(CPU only).

## Contents
1. Install and model
2. What the encoders produce
3. Indexing and retrieval with PISA
4. Indexing and retrieval with Terrier
5. Prebuilt SPLADE vectors
6. Problems

## 1. Install and model

```bash
pip install pyterrier-splade      # import pyterrier_splade (the old package name pyt_splade is legacy)
```

```python
import pyterrier_splade
splade = pyterrier_splade.Splade()   # default model: naver/splade-cocondenser-ensembledistil
```

`Splade(model="naver/splade-cocondenser-ensembledistil", tokenizer=None, agg="max", max_length=256,
device=None)`. Pass another checkpoint name for other SPLADE versions, and use the same one for
documents and queries.

## 2. What the encoders produce

| Transformer | Input | Adds |
|---|---|---|
| `splade.doc_encoder()` | documents with `text` | `toks`: dict of wordpiece to integer weight, e.g. `{'compact': 257, 'memories': 177}` |
| `splade.query_encoder()` | queries with `query` | `query_toks`: wordpiece to float weight (×100), for the query |
| `splade.scorer(text_field="text", batch_size=100)` | results with `query` and `text` | a SPLADE score at scale 1 (the raw dot product), to rerank without an index |

The terms are wordpieces from SPLADE's vocabulary, not words. They must not be stemmed or stopped again.

## 3. Indexing and retrieval with PISA

This is the faster route for large corpora.

```python
from pyterrier_pisa import PisaIndex

index = PisaIndex("collection.splade.pisa", stemmer="none")
(splade.doc_encoder() >> index.toks_indexer()).index(dataset.get_corpus_iter())
retriever = splade.query_encoder() >> index.quantized()
```

- `quantized()` scores with the stored impact values. Don't use `bm25()` on a SPLADE index.
- Calling `quantized()` builds a `quantized.*` structure inside the index folder.
- Scales stack: `doc_encoder(scale=100)` already emits integer weights ×100, and PISA's
  `toks_indexer(scale=100.0)` multiplies and truncates them again; `query_encoder(scale=100)` ×
  `quantized(toks_scale=100)` does the same for queries. Rankings are unaffected, but scores are about 1e8 × the raw dot
  product (1e8–1e9), so `scorer()` and `quantized()` scores can't be compared or mixed.
- PISA's `toks_indexer()` raises `ValueError: To index from dicts, you must set stemmer='none'` unless the
  index was created with `stemmer="none"`. To rebuild, use `toks_indexer(mode="overwrite")`.

## 4. Indexing and retrieval with Terrier

```python
import os
import pyterrier as pt

index = pt.terrier.TerrierIndex("collection.splade.terrier")
(splade.doc_encoder() >> index.toks_indexer()).index(dataset.get_corpus_iter())
retriever = splade.query_encoder() >> index.retriever("Tf")    # Tf: sum of stored weights
```

With `IterDictIndexer`, use `pretokenised=True` and an absolute path:

```python
indexer = pt.IterDictIndexer(os.path.abspath("collection.splade.terrier"), pretokenised=True)
index_ref = (splade.doc_encoder() >> indexer).index(dataset.get_corpus_iter())
retriever = splade.query_encoder() >> pt.terrier.Retriever(index_ref, wmodel="Tf")
```

Use `wmodel="Tf"`, so the score is the sum of the SPLADE weights. Other models such as BM25 reweight
the terms and break the model.

## 5. Prebuilt SPLADE vectors

Encoding a corpus with SPLADE needs a GPU for anything but small collections. Prebuilt document
vectors exist on Hugging Face as indexer caches, e.g. `pyterrier/<beir-corpus>.splade-v3.cache`
and `macavaney/msmarco-passage.splade-lg.cache` (see the `pyterrier-huggingface` skill). They hold
`toks` for every document, so you only need to index them. Check each card for the exact model, and
encode queries with that same model. They load as pyterrier-caching indexer caches:

```python
from pyterrier_caching import IndexerCache                          # pip install pyterrier-caching
cache = IndexerCache.from_hf("macavaney/vaswani.splade.cache")      # card: built with the default Splade()
PisaIndex("vaswani.splade.pisa", stemmer="none").toks_indexer().index(iter(cache))
```

## 6. Problems

| Symptom | Cause | Fix |
|---|---|---|
| `JavaException … Cannot create new index` | `IterDictIndexer` given a relative path | `os.path.abspath(path)` |

Other symptoms (legacy import, stemmed index, wrong Terrier model, checkpoint mismatch, CPU encoding) are
in the SKILL.md Troubleshooting table.
