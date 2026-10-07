# Custom transformers

All code here was checked with PyTerrier 1.1.2.

## Contents
1. Choosing `transform` or `transform_iter`
2. Model-backed reranker (batched)
3. Query rewriter
4. Row-by-row transformer with `transform_iter`
5. Working with `pt.inspect`
6. Checklist

## 1. Choosing `transform` or `transform_iter`

| Implement | When |
|---|---|
| `transform(self, df)` | most cases; you work with a whole DataFrame, often grouped by `qid` |
| `transform_iter(self, rows)` | the logic is naturally one dict per row, with no grouping |

Implement only one. PyTerrier provides the other, as well as `.search()`.

## 2. Model-backed reranker (batched)

Load the model once in `__init__`. Score in batches in `transform`. Keep every input column and
replace `score`, then recompute ranks.

```python
import pyterrier as pt

class BatchedScorer(pt.Transformer):
    def __init__(self, model, batch_size=32):
        self.model = model                 # loaded once, e.g. a CrossEncoder
        self.batch_size = batch_size

    def transform(self, inp):
        pt.validate.result_frame(inp, extra_columns=["text"])
        if len(inp) == 0:                  # pt.inspect may call with an empty frame
            return inp.assign(score=[], rank=[])
        scores = []
        for start in range(0, len(inp), self.batch_size):
            batch = inp.iloc[start:start + self.batch_size]
            pairs = list(zip(batch["query"], batch["text"]))
            scores.extend(self.model.predict(pairs))
        return pt.model.add_ranks(inp.assign(score=scores))
```

- `pt.apply.doc_score(fn, batch_size=...)` does the same thing when you don't need a class.
- To keep the reranker's memory bounded, put a cut-off in front of it (`first_stage % k >> reranker`).

## 3. Query rewriter

Keep the original query in `query_0` with `pt.model.push_queries(..., keep_original=True)`. Without
`keep_original=True`, `push_queries` moves `query` to `query_0` and the frame no longer has a `query`
column.

```python
class AddTerm(pt.Transformer):
    def __init__(self, term):
        self.term = term

    def transform(self, inp):
        pt.validate.query_frame(inp)
        out = pt.model.push_queries(inp, keep_original=True)   # query_0 = previous query
        return out.assign(query=out["query"] + " " + self.term)

# Retrieve with the rewritten query, then restore the original for later stages
pipeline = AddTerm("synthesis") >> bm25 >> pt.rewrite.reset()
```

Restore the original query with `pt.rewrite.reset()` before a neural reranker, so the reranker sees
what the user typed.

## 4. Row-by-row transformer with `transform_iter`

```python
class UpperCaseQuery(pt.Transformer):
    def transform_iter(self, inp):
        for row in inp:                    # each row is a dict
            yield {**row, "query": row["query"].upper()}
```

## 5. Working with `pt.inspect`

- `pt.inspect.transformer_outputs(pipeline, ["qid", "query"])` reports the columns a pipeline will
  output, without running it on real data.
- To do that, PyTerrier may call your transformer with an **empty DataFrame**. Make sure an empty
  input returns an empty frame with the right columns, as in section 2. Don't call the model on it.
- Implement `transform_outputs(self, input_columns)` only when calling the transformer is expensive
  even for an empty input. It must return exactly the columns that `transform` would produce.

## 6. Checklist

- [ ] Validates its input columns (`pt.validate.result_frame` or `pt.validate.query_frame`).
- [ ] Handles an empty input frame.
- [ ] Returns `qid` and `docno` as strings, and keeps the input columns later stages may need.
- [ ] Recomputes ranks with `pt.model.add_ranks` after changing scores.
- [ ] Loads models once and scores in batches.
- [ ] Imports nothing from legacy APIs (`TransformerBase`, `pta.DataFrameBuilder`).
