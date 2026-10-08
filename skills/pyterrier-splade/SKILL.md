---
name: pyterrier-splade
description: >-
  Uses SPLADE learned sparse models in PyTerrier with pyterrier-splade, in
  every role: encoding documents and queries into weighted wordpiece terms,
  indexing them in PISA (toks_indexer, quantized) or Terrier (toks_indexer,
  Tf), retrieving with SPLADE, reranking any first stage with the index-free
  splade.scorer(), and indexing prebuilt SPLADE vectors (splade-v3,
  splade-lg caches) from Hugging Face. Covers checkpoints, stemming and
  weighting traps, and replacing the legacy pyt_splade package. Use whenever
  SPLADE or learned sparse retrieval appears in a PyTerrier task, whether as
  retriever or reranker. Not for BM25 on PISA or Terrier, or for fusing SPLADE
  with other retrievers (see pyterrier-pipelines).
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-splade 0.1.0, pyterrier-pisa 0.4.7, CPU only)
---

# SPLADE (learned sparse retrieval)

Assumes the `pyterrier-basics` skill.

Docs: [SPLADE](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-splade/index.html) ·
[PISA](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier_pisa/index.html) ·
[Terrier indexing](https://pyterrier.readthedocs.io/en/latest/terrier/indexing.html)

## Choosing a SPLADE role

| Role | Use | Needs |
|---|---|---|
| Retrieve, large corpus | `splade.query_encoder() >> pisa_index.quantized()` | a PISA index of SPLADE `toks` (`pyterrier-pisa`, Linux x86_64) |
| Retrieve, any platform | `splade.query_encoder() >> terrier_index.retriever("Tf")` | a Terrier index of SPLADE `toks` |
| Rerank another first stage | `splade.scorer()` | only the checkpoint and document text |
| Skip encoding the corpus | prebuilt `*.splade*.cache` from Hugging Face | the same checkpoint for queries |

Check for prebuilt vectors or indexes first (see the `pyterrier-huggingface` skill). Encoding a corpus
needs a GPU for anything but small collections.

## Install

```bash
pip install pyterrier-splade pyterrier-pisa    # import pyterrier_splade; pyt_splade is legacy
```

```python run slow
import itertools
import pyterrier as pt
import pyterrier_splade

dataset = pt.get_dataset("irds:vaswani")
docs = list(itertools.islice(dataset.get_corpus_iter(), 50))   # small slice: CPU encoding is slow
splade = pyterrier_splade.Splade()        # checkpoint stated in references/splade.md; pass model= to change it
```

Use the same checkpoint for documents and queries.

## Index and retrieve

```python run slow
from pyterrier_pisa import PisaIndex

sparse_index = PisaIndex("vaswani.splade.pisa", stemmer="none")
if not sparse_index.built():
    (splade.doc_encoder() >> sparse_index.toks_indexer()).index(docs)
splade_retriever = splade.query_encoder() >> sparse_index.quantized()
assert len(splade_retriever.search("chemical reactions")) > 0
```

- `doc_encoder()` adds a `toks` column (wordpiece → integer weight); `query_encoder()` adds `query_toks`.
  The terms are wordpieces, so they must not be stemmed or stopped again.
- `quantized()` scores with the stored weights. Never put `bm25()` on a SPLADE index.
- For the Terrier variant (`toks_indexer`, `retriever("Tf")`), read [references/splade.md](references/splade.md).

## Rerank

```python run slow
bm25 = pt.terrier.TerrierIndex.example().bm25(k1=1.2, b=0.75)
pipeline = bm25 % 10 >> dataset.text_loader() >> splade.scorer()
reranked = pipeline.search("chemical reactions")
assert len(reranked) == 10
```

- `scorer(text_field="text", batch_size=100)` encodes each query once and every document on the fly, then
  scores by dot product. It needs `query` and the text column.
- The rerank depth (`% 10`) comes from the task. For choosing it and evaluating the reranker, see the
  `pyterrier-pipelines` skill.

## Prebuilt vectors

Prebuilt document vectors exist as indexer caches, e.g. `pyterrier/<beir-corpus>.splade-v3.cache` and
`macavaney/msmarco-passage.splade-lg.cache`. They hold `toks` for every document, so only indexing is
left. Check each card for the checkpoint and encode queries with it. For loading them, read
[references/splade.md](references/splade.md).

## Don't

- Don't `import pyt_splade`. Use `pyterrier_splade`.
- Don't stem or stop SPLADE terms. Build a PISA index with `stemmer="none"`, or use `toks_indexer`.
- Don't score a SPLADE index with BM25. Use `quantized()` on PISA or `"Tf"` on Terrier.
- Don't query an index with a different SPLADE checkpoint from the one that built it.
- Don't call `text_loader()` on a PISA index. It stores no text; use `dataset.text_loader()`.
- Don't encode a large corpus on CPU. Use a GPU or a prebuilt cache.

## Reference files

| Read | When |
|---|---|
| [references/splade.md](references/splade.md) | choosing a checkpoint, what the encoders produce, Terrier indexing, prebuilt SPLADE vectors |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pyt_splade` | legacy package name | `pip install pyterrier-splade`, `import pyterrier_splade` |
| Poor results after indexing | index built with stemming or stopwords | rebuild with `stemmer="none"` (PISA) or via `toks_indexer` |
| Poor results with Terrier | retriever is not `Tf` | `index.retriever("Tf")` or `wmodel="Tf"` |
| Poor results with a prebuilt cache | queries encoded with a different checkpoint | use the checkpoint named on the card |
| `InputValidationError` naming `text` from `scorer()` | no text loader before the scorer | `dataset.text_loader()` |
| `RuntimeError: A PISA index already exists at …` | indexing into an existing PISA index | check `built()`, or `PisaIndex(path, overwrite=True)` |
| Encoding takes hours | CPU only | GPU, or a prebuilt `.splade*.cache` |
