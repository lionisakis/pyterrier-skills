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

- `text_field`: the one field to index. If it's left out, every `str` field is indexed and a warning is
  printed. Set it. The prebuilt BEIR indexes on Hugging Face index title and text together, so to match
  them concatenate first: `({"docno": d["docno"], "text": f'{d["title"]} {d["text"]}'} for d in docs)`.
- `stemmer`: `porter2` (default), `krovetz` or `"none"`. PISA has **no** Porter v1, which is Terrier's
  default. To match stemming across backends, build the Terrier index with
  `stemmer="EnglishSnowballStemmer"` (Porter2; Terrier accepts a Java class name), or report that the
  stemmers differ. A prebuilt Terrier index can't be changed: read its stemmer from the `termpipelines`
  line of `{index.path}/data.properties` (`pyterrier/vaswani.terrier` has `Stopwords,PorterStemmer`,
  i.e. Porter v1). If it's Porter v1, rebuild the Terrier index yourself with Porter2 or report the mismatch. Use `"none"` for learned sparse or non-English text. Even with k1, b, stemmer and
  stopwords matched, the tokenisers still differ (PISA exposes no tokeniser option), so scores and top-10s
  diverge (nfcorpus, k1=0.9 b=0.4, Porter2 both sides, 308 test queries: scores ≈ 0.70 ×; top-10
  overlap 0.84 as the mean over queries of shared docnos ÷ 10, or 0.95 as mean Jaccard). Say which
  overlap you report. Compare by rank, not by score.
- `stops`: the Terrier English stopword list by default. Pass `"none"` or a list to change it. Keep it
  matched with Terrier: `stopwords="none"` on one side only widens the divergence sharply (scores ≈ 0.27 ×).
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
| `index.quantized()` | for learned sparse indexes (see the `pyterrier-splade` skill) |

- Each scoring setting builds its own compressed structures on first use, e.g.
  `bm25.k1-0.9.b-0.4.q0.bmw.64.block_simdbp` inside the index folder. The first query with a new
  setting is slower, and every setting you try takes disk space.
- PISA results have no `docid` column, only `docno`.
- PISA tokenises the raw query itself and splits on punctuation, so queries with `:` need no stripping.
  Only the Terrier side does (references/terrier.md).
- With default settings on the same corpus, PISA BM25 and Terrier BM25 give different scores, because
  their stemmer and k1/b both differ. On vaswani, nDCG@10 was 0.467 for PISA and 0.447 for Terrier. Pass
  matching parameters when you compare backends.

## 4. Learned sparse indexes

Pre-weighted terms (SPLADE and similar) go in with `toks_indexer()` on an index created with
`stemmer="none"` and are queried with `quantized()`. See the `pyterrier-splade` skill.

## 5. Errors

| Message | Cause | Fix |
|---|---|---|
| `RuntimeError: A PISA index already exists at … set mode="overwrite"` | indexing into an existing index | new path, `PisaIndex(path, overwrite=True)`, or `indexer(mode="overwrite")` |
| `UserWarning: text_field not specified; indexing all str fields` | no `text_field` | `PisaIndex(path, text_field="text")` |
| `pip install` finds no matching distribution | not Linux x86_64 | use Terrier |
| `AttributeError: … 'text_loader'` | PISA stores no text | use `dataset.text_loader()` |
