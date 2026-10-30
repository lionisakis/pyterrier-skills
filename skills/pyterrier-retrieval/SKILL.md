---
name: pyterrier-retrieval
description: >-
  Builds and queries lexical first-stage retrievers in PyTerrier: Terrier
  indexes and BM25/DPH/PL2 retrieval, PISA indexes, retriever parameters,
  metadata lengths, tokenisation and stemming, query punctuation, and the
  comparison table for choosing a first-stage family (lexical, SPLADE, dense,
  ColBERT). Use when indexing a corpus for BM25, choosing or configuring a
  first-stage retriever, or comparing Terrier with PISA in PyTerrier. Not for
  reranking, query expansion, evaluation, or fusing rankings (see
  pyterrier-fusion). For SPLADE, dense (pyterrier-dr) or ColBERT models, use
  pyterrier-splade, pyterrier-dense or pyterrier-colbert.
compatibility: pyterrier>=1.1 (checked with 1.1.2, pyterrier-pisa 0.4.7)
---

# First-stage retrieval

Assumes the `pyterrier-basics` skill.

Docs: [Terrier indexing](https://pyterrier.readthedocs.io/en/latest/terrier/indexing.html) ·
[Terrier retrieval](https://pyterrier.readthedocs.io/en/latest/terrier/retrieval.html) ·
[PISA](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier_pisa/index.html)

## Choosing a first stage

Check for a prebuilt index first (see the `pyterrier-huggingface` skill). Build one only if none fits.

| Family | Default choice | Needs | CPU cost | When it wins | Read |
|---|---|---|---|---|---|
| Lexical (BM25, DPH, PL2) | Terrier | core, Java 11+ | low | the baseline every paper reports; exact term match; any corpus size | [references/terrier.md](references/terrier.md) |
| Lexical, fastest queries | PISA | `pyterrier-pisa`, Linux x86_64, no Java | low; fastest query time | tens of millions of documents, latency experiments | [references/pisa.md](references/pisa.md) |
| Learned sparse | SPLADE in PISA or Terrier | `pyterrier-splade`; a GPU to encode the corpus, or a prebuilt `.splade*.cache` | encoding slow, querying at lexical speed | vocabulary mismatch while keeping an inverted index | the `pyterrier-splade` skill |
| Dense (bi-encoder) | pyterrier-dr `FlexIndex` | `pyterrier-dr`; a GPU to encode, or a prebuilt `.flex` | encoding slow; exact search fine up to a few million vectors | semantic match and paraphrase, models trained in domain | the `pyterrier-dense` skill |
| Late interaction | ColBERT v2 / PLAID, or PyLate `PlaidIndex` | `pyterrier_colbert2` (transformers <5, GPU recommended) or `pyterrier-pylate` (transformers <=5.3); one environment each | encoding slow, large index | strongest zero-shot first stage; token-level matching | the `pyterrier-colbert` skill |
| Two or more rankings | reciprocal rank fusion | `pyterrier-alpha` | negligible | sparse and dense complement each other | the `pyterrier-fusion` skill |

Every learned family also works as a reranker without an index; the family skills cover both roles.
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
pisa_bm25 = pisa.bm25(k1=1.2, b=0.75)            # PISA's own defaults differ: see references/defaults.md
assert len(pisa_bm25.search("chemical reactions")) > 0
```

PISA stores no text and has no `text_loader()`. Use `dataset.text_loader()` before a reranker.

## Combining retrievers

Fuse two rankings by rank with `pta.RRFusion(bm25, dense, k=60)`, or normalise scores and add them. Read
the `pyterrier-fusion` skill; it also covers fusing rerankers and saved runs.

## Parameters

Never rely on default parameters. They differ between backends: BM25 defaults differ between Terrier
and PISA, as do their stemmers. Pass every parameter explicitly, and check the values actually used
against the installed source, the index metadata or the paper. For indicative values and how to check
them, read [references/defaults.md](references/defaults.md). To tune them, use
`pt.terrier.Retriever(index, wmodel="BM25", controls={...})` (`index.bm25()` can't be tuned) and the
`pyterrier-experiments` skill.

## Don't

- Don't build an index when a suitable prebuilt one exists.
- Don't create `pt.terrier.Retriever(index)` without `wmodel=`. The default is DPH.
- Don't index a real collection with `TerrierIndex(path).index(...)`'s guessed metadata lengths.
- Don't give `pt.IterDictIndexer` a relative path. Use `os.path.abspath(path)`, or `TerrierIndex`.
- Don't expect to add documents to an existing Terrier or PISA index. Rebuild it.
- Don't add raw scores from different retrievers. Fuse by rank or normalise first (the `pyterrier-fusion` skill).
- Don't call `text_loader()` on a PISA index. It stores no text.
- Don't compare Terrier and PISA (or Anserini) BM25 without matching k1, b, stemmer and stopwords (same `k1=`,
  `b=` in both `bm25()` calls; stemmers in references/pisa.md). Even matched, their tokenisers differ and so do
  the scores, so compare by rank, not by score.
- Don't send a query whose only punctuation is `:` to a Terrier retriever. Other punctuation makes PyTerrier
  tokenise the query, `:` alone doesn't, and on an index built without fields, the default, such a query
  aborts the whole run. Strip punctuation, or replace only `:` when queries must stay raw (references/terrier.md).
- Don't pass `threads>1` to a Terrier retriever (`index.bm25(threads=8)`). Multi-threaded retrieval is
  experimental and fails intermittently; after one failure, later retrievals in the process fail too.

## Reference files

| Read | When |
|---|---|
| [references/terrier.md](references/terrier.md) | building or querying a Terrier index: metadata, tokenisation, stemming, fields, positions, errors |
| [references/pisa.md](references/pisa.md) | building or querying a PISA index for BM25 or another lexical model |
| [references/defaults.md](references/defaults.md) | choosing or reporting BM25 and other parameters |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: an index is already built at this path` | `TerrierIndex(path).index(...)` on an existing index | check `index.built()` first, or use a new path |
| `NullPointerException … "this.val$tpa" is null`, or `Unknown problem looking for doclength for document N` | multi-threaded Terrier retrieval (`threads>1`) | `threads=1` (the default) and restart the Python process |
| `RuntimeError: A PISA index already exists at …` | indexing into an existing PISA index | `PisaIndex(path, overwrite=True)` or a new path |
| `JavaException … Cannot create new index: path …/var/<name> does not exist` | `IterDictIndexer` given a relative path | `os.path.abspath(path)` |
| Java log `Term … written in N bytes. Max expected size was M`; docnos or text cut short | metadata wider than its `meta` width | rebuild with larger `meta` widths |
| BM25 scores differ between backends on the same data | different k1, b, stemmer, stopwords, tokeniser and BM25 formula | match k1, b, stemmer and stopwords (references/pisa.md); the scores still differ and the top-10s only partly overlap (nfcorpus, k1=0.9 b=0.4, Porter2 on both sides: PISA ≈ 0.70 × Terrier, top-10 overlap 0.84), so compare by rank |
| Java ERROR `Unknown field X - known fields are [text]`, `returning empty result set as query …`; a topic gets no results, and in `pt.Experiment` that system's significance columns are NaN | the query contains `word:` (TerrierQL field syntax), which isn't auto-tokenised; the index has fields, so Terrier looks the "field" up | strip punctuation before the retriever (references/terrier.md section 3) |
| `jnius.JavaException … IllegalStateException: Index must have 1 or more fields` aborts the whole run | the same `word:` query against an index built without fields (the default) | strip punctuation before the retriever (references/terrier.md section 3) |
| `KeyError: 'text'` in the next stage | the index stores no text | `dataset.text_loader()` |
| `pip install pyterrier-pisa` finds no distribution | not Linux x86_64 | use Terrier |
