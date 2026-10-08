# Terrier indexing and retrieval

Checked with PyTerrier 1.1.2 and Terrier 5.11.

## Contents
1. Building an index
2. Metadata lengths and truncation
3. Retrieval
4. Tokenisation, stemming and stopwords
5. Fields and positions
6. Errors

## 1. Building an index

| Situation | Use |
|---|---|
| Quick index, small or uniform corpus | `pt.terrier.TerrierIndex(path).index(docs)` |
| Real collection: control the metadata, text fields or language | `pt.terrier.TerrierIndex(path).indexer(meta=..., text_attrs=[...], ...).index(docs)` |
| Pre-tokenised input (e.g. SPLADE `toks`) | `pt.terrier.TerrierIndex(path).toks_indexer(meta=...).index(docs)` |
| Older code using `IterDictIndexer` | `pt.IterDictIndexer(os.path.abspath(path), meta=..., ...)`. Give it an absolute path |

```python
import pyterrier as pt

dataset = pt.get_dataset("irds:vaswani")
index = pt.terrier.TerrierIndex("vaswani.terrier")
index.indexer(
    meta={"docno": 20, "text": 4096},   # store docno and text, with maximum lengths
    text_attrs=["text"],                 # fields to index
    threads=4,
).index(dataset.get_corpus_iter())
```

- `docs` is any iterable of dicts with a `docno` and the text fields, usually `dataset.get_corpus_iter()`.
- `TerrierIndex.indexer(...)` signature: `meta={'docno': 20}`, `text_attrs=['text']`,
  `tokeniser='english'`, `stemmer='porter'`, `stopwords='terrier'`, `store_separate_fields=False`,
  `store_positions=False`, `threads=1`.
- A Terrier index can't be extended. Re-index from scratch to add documents.
- Using an index built earlier: `pt.terrier.TerrierIndex(path)`, then `.built()` to check it exists.

## 2. Metadata lengths and truncation

Terrier stores metadata (docno, text, title, ...) in fixed-width fields. A longer value is **cut off**:
Java logs an error ("Term … written in N bytes. Max expected size was M") but indexing carries on.

- `TerrierIndex(path).index(docs)` guesses the widths from the **first document only**: 10 × the
  docno length (at least 20) and 10 × each text field's length (at least 100). One short first
  document means truncated docnos and text for the rest. Use `.indexer(meta=...)` for real collections.
- Pick `meta` widths from the corpus: the longest docno, and enough text for rerankers (e.g. 4096
  characters for passages). Text that isn't in `meta` can't be loaded later with `index.text_loader()`.
- Common docno lengths: MS MARCO passage ids fit in 20. Other collections can be much longer, so check
  `max(len(d["docno"]) for d in docs)` on a sample.

## 3. Retrieval

```python
bm25 = index.bm25(k1=1.2, b=0.75, num_results=1000)          # explicit parameters
dph = index.dph()
pl2 = index.pl2()
tf = index.retriever("Tf")                                     # any Terrier weighting model
tunable = pt.terrier.Retriever(index, wmodel="BM25", controls={"bm25.b": 0.75})  # for tuning
```

- `TerrierIndex` shortcuts: `bm25`, `dph`, `pl2`, `tf`, `tf_idf`, `dirichlet_lm`, `hiemstra_lm`,
  `retriever(model, model_args)`, plus query expansion (`rm3`, `bo1`, `kl`, `sdm`; see the
  `pyterrier-expansion` skill).
- `pt.terrier.Retriever(index)` **without** `wmodel=` uses DPH. Always pass `wmodel`.
- Create the index object once and reuse it for every retriever in a script.
- Results include Terrier's integer `docid` as well as `docno`.
- Queries are tokenised automatically, so punctuation in queries no longer breaks parsing.

## 4. Tokenisation, stemming and stopwords

| Option | Values |
|---|---|
| `tokeniser` | `english` (default), `utf`, `whitespace`, `twitter`, `identity` |
| `stemmer` | `porter` (default), `weakporter`, `none`, and Snowball stemmers: `danish`, `finnish`, `german`, `hungarian`, `norwegian`, `portugese`, `spanish`, `swedish`, `turkish`, ... |
| `stopwords` | `terrier` (default, English), `none`, or a list of words |

- For non-English text use `tokeniser="utf"`, the matching stemmer or `"none"`, and
  `stopwords="none"` (the default list is English).
- The retriever uses the index's own settings, so queries are processed the same way as documents.

## 5. Fields and positions

- `store_separate_fields=True` keeps per-field statistics, needed for field-based models such as
  BM25F and for `include_fields=` in `bm25(...)`. It's off by default.
- `store_positions=True` stores term positions, needed for proximity models such as SDM. It's off by
  default.
- Both make the index larger. Turn them on only when the model needs them.

## 6. Errors

| Message | Cause | Fix |
|---|---|---|
| `AssertionError: an index is already built at this path` | `TerrierIndex(path).index(...)` on an existing index | use a new path, or delete the old index |
| `ValueError: Index already exists at …/data.properties` | `IterDictIndexer` on an existing index | `overwrite=True`, or a new path |
| `JavaException … Cannot create new index: path …/var/<name> does not exist` | `IterDictIndexer` given a relative path | pass `os.path.abspath(path)`, or use `TerrierIndex(path)` |
| Java log `Term … written in N bytes. Max expected size was M` | a metadata value longer than its `meta` width | rebuild with larger `meta` widths |
| `AssertionError: input documents must contain a 'docno' field` | documents lack `docno` | rename the id field (e.g. `doc_id` to `docno`) |
