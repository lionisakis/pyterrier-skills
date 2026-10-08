# doc2query and Doc2Query--

Checked with pyterrier-doc2query (0.2.0 from PyPI, and GitHub master for Doc2Query--), transformers 5
and PyTerrier 1.1.2, on CPU over two vaswani documents.

## Contents
1. Install
2. Generating queries
3. Doc2Query--: scoring and filtering
4. Indexing and caching
5. Known problems

## 1. Install

```bash
pip install pyterrier-doc2query protobuf
# Doc2Query-- (QueryScorer) needs the GitHub version until a release after 0.2.0:
pip install git+https://github.com/terrierteam/pyterrier_doc2query.git
```

- The GitHub version also reports `0.2.0`, so the version number doesn't tell you whether you have
  the fix. Check for `'docno': []` in `pyterrier_doc2query/filtering.py`, or just run `QueryScorer` on
  one document.
- `protobuf` is required with transformers 5. The default checkpoint ships only a SentencePiece file
  (`spiece.model`). Without `protobuf`, loading the tokenizer fails with the misleading
  `ValueError: Error parsing line b'\x0e' in …/spiece.model`, which comes from the tiktoken parser.
- A GPU is needed for any real corpus: every document runs `num_samples` T5 generations.

## 2. Generating queries

```python
from pyterrier_doc2query import Doc2Query

doc2query = Doc2Query(num_samples=40, batch_size=16)
expanded = doc2query(documents_df)          # adds a `querygen` column, one query per line
```

`Doc2Query(checkpoint="macavaney/doc2query-t5-base-msmarco", num_samples=3, batch_size=4,
doc_attr="text", append=False, out_attr="querygen", fast_tokenizer=False, device=None)`.

| `append` | Result |
|---|---|
| `False` (default) | queries go into `querygen` as one newline-separated string; `text` is unchanged |
| `True` | queries are appended to `text`; there is no `querygen` column |

- Use `append=False` when the queries will be scored and filtered (section 3), and `append=True`
  when they go straight into an index.
- `fast_tokenizer=True` speeds up generation (the package warns if it's off).
- The number of samples is the main quality and cost setting. The Doc2Query and Doc2Query-- papers
  use tens of samples per document (e.g. 40 or 80).

## 3. Doc2Query--: scoring and filtering

Doc2Query-- scores each generated query against its document with a relevance model, and keeps only
the queries scoring above a threshold:

```python
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
import pyterrier_dr

pipeline = (
    Doc2Query(num_samples=40, append=False)
    >> QueryScorer(pyterrier_dr.ElectraScorer())     # adds `querygen_score`, one score per query
    >> QueryFilter(t=threshold, append=True)          # keeps queries with score >= t, appends them to `text`
)
```

- `QueryScorer(scorer)` takes any query-document scorer, e.g. monoELECTRA or a bi-encoder from
  pyterrier-dr.
- `QueryFilter(t, append=True)` keeps queries scoring **≥ t**, appends them to `text`, and drops the
  `querygen` columns. With `append=False`, it filters `querygen` and `querygen_score` instead.
- `t` is on the scorer's own scale, so it differs for every scorer. Doc2Query-- chooses it as a
  percentile of the query scores over a sample of the corpus (e.g. keep the top 30%). Compute it on a
  sample, rather than reusing a number from another setup.
- Prebuilt Doc2Query-- indexes for MS MARCO passage exist as `pt.Artifact.from_dataset("msmarco_passage",
  "pisa_d2qmm_n40_electra_p70")` and similar variants (see the `pyterrier-huggingface` skill).

## 4. Indexing and caching

```python
from pyterrier_caching import IndexerCache

cache = IndexerCache("corpus.d2q.cache")
(Doc2Query(num_samples=40, append=False) >> cache).index(dataset.get_corpus_iter())    # once

for t in [low, high]:      # try thresholds without regenerating
    index = pt.terrier.TerrierIndex(f"corpus.d2qmm-{t}.terrier")
    (QueryScorer(scorer) >> QueryFilter(t=t) >> index.indexer(meta={"docno": 20})).index(iter(cache))
```

- Generation is by far the slowest step, so cache its output and vary only the filtering.
- `Doc2QueryStore` and `QueryScoreStore` in this package are deprecated in favour of
  `pyterrier_caching.IndexerCache` (see the `pyterrier-caching` skill).

## 5. Known problems

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Error parsing line b'\x0e' in …/spiece.model` | transformers 5 without `protobuf` | `pip install protobuf` |
| `KeyError: 'docno'` from `QueryScorer` | bug in pyterrier-doc2query 0.2.0 (PyPI) | install from GitHub master |
| Indexed documents look unexpanded | `append=False` and no `QueryFilter(append=True)` | set `append=True` on `Doc2Query`, or add the filter |
| Generation takes days | CPU, or too many samples | GPU; cache with `IndexerCache`; use a prebuilt index |
