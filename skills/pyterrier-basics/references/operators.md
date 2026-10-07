# Operator precedence, worked examples

PyTerrier operators are ordinary Python operators, so Python's precedence decides how a pipeline is
grouped. Highest first: `**`, then `*` and `%`, then `+`, then `>>`, then `&`, then `^`, then `|`.

Every row below was checked with PyTerrier 1.1.2 by printing the pipeline. `print(pipe)` shows the
grouping of any pipeline.

| You write | Python groups it as | Usually meant | Fix |
|---|---|---|---|
| `bm25 % 10 >> tf` | `(bm25 % 10) >> tf` | the same | none needed |
| `bm25 >> tf % 10` | `bm25 >> (tf % 10)` | the same: cut off after tf | none needed, but `(bm25 >> tf) % 10` is clearer |
| `bm25 + 2 * dph` | `bm25 + (2 * dph)` | the same | none needed |
| `bm25 >> tf + pl2` | `bm25 >> (tf + pl2)` | the same: rescore with tf+pl2 | parenthesise anyway |
| `bm25 >> tf ** pl2` | `bm25 >> (tf ** pl2)` | the same: features from tf and pl2 | parenthesise anyway |
| `bm25 >> tf \| pl2` | `(bm25 >> tf) \| pl2` | `bm25 >> (tf \| pl2)` | parenthesise the union |
| `bm25 & dph >> tf` | `bm25 & (dph >> tf)` | `(bm25 & dph) >> tf` | parenthesise the intersection |
| `bm25 % 10 ^ dph` | `(bm25 % 10) ^ dph` | the same | none needed |

## Things operators do that are easy to miss

- `|` and `&` return documents only, with no `score` or `rank` columns. Rescore afterwards, e.g.
  `(bm25 | dph) >> reranker`.
- `+` returns the union of both sides' documents, and a document missing from one side gets 0 from
  that side. The two sides' score ranges are not normalised,
  so a weight such as `bm25 + 0.1 * dense` depends on the score scale. Normalise first (see the
  `pyterrier-first-stage` skill on fusion).
- `**` keeps the left-hand input's score and adds a `features` numpy array, with one entry per
  transformer, in order. `bm25 >> (tf ** pl2)` gives `features = [tf, pl2]`.
- `%` applies per query and keeps rank order.
- `^` keeps every result from the left side, then appends results from the right side that are not
  already present.

## Writing long pipelines

Break long pipelines over lines inside parentheses, one stage per line:

```python
pipeline = (
    first_stage % 100
    >> text_loader
    >> reranker
)
```
