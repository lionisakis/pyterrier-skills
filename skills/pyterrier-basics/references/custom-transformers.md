# Custom transformers

All code here was checked with PyTerrier 1.1.2 and pyterrier-dr 0.8.0.

## Contents
1. Choosing `transform` or `transform_iter`
2. The input and output contract
3. Model-backed reranker (batched)
4. Query rewriter
5. One transformer, several modes
6. Row-by-row transformer with `transform_iter`
7. `pt.apply` transformers and inspection
8. Checking the contract
9. Checklist

## 1. Choosing `transform` or `transform_iter`

| Implement | When |
|---|---|
| `transform(self, df)` | most cases; you work with a whole DataFrame, often grouped by `qid` |
| `transform_iter(self, rows)` | the logic is naturally one dict per row, with no grouping |

Implement only one. PyTerrier provides the other, as well as `.search()`. A `transform_iter`
transformer needs extra work to be inspectable (section 6).

## 2. The input and output contract

PyTerrier never reads a declaration of what a transformer consumes and produces. It runs the
transformer to find out, so the code has to be written so that the answer is cheap and correct:

| Question | How PyTerrier answers it (`pt.inspect`) | What your transformer must do |
|---|---|---|
| Which input columns does `t` accept? | `transformer_inputs(t)` calls `t` with a frame that has **no columns** and reads `missing_columns` from the `InputValidationError`. If `t` defines `transform_inputs` (method or attribute returning `List[str]` or `List[List[str]]`), that is used instead | validate as the first statement of `transform`, before touching any column |
| Which columns does `t` output for input columns `C`? | `transformer_outputs(t, C)` calls `t.transform` with an **empty** frame whose columns are `C` and returns the result's columns. If `t` defines `transform_outputs(input_columns)`, that is called instead | return an empty frame with exactly the output columns, without calling the model |

Who uses the answers:

- `pt.Experiment` validates every system against the topics' columns before running anything
  (`validate="warn"` by default, `"error"` to stop, `"ignore"` to skip). A pipeline whose stages don't
  fit is reported as "failed validation"; a stage whose contract can't be determined as "could not be validated".
- Composed pipelines (`a >> b`) implement `transform_outputs` by chaining the stages, so the whole
  pipeline is checkable only when every stage is.
- Schematics (printing a pipeline in a notebook) show the columns flowing between stages, or `?`
  for a stage that can't be inspected.

### Validation functions

| Function | Must have | Must not have |
|---|---|---|
| `pt.validate.query_frame(inp, extra_columns=[...])` | `qid` + extras | `docno` |
| `pt.validate.result_frame(inp, extra_columns=[...])` | `qid`, `docno` + extras | |
| `pt.validate.document_frame(inp, extra_columns=[...])` | `docno` + extras | `qid` |
| `pt.validate.columns(inp, includes=[...], excludes=[...])` | `includes` | `excludes` |
| `pt.validate.any(inp) as v` | one of several specs (section 5) | |
| `*_iter` variants (`query_iter`, `result_iter`, `document_iter`, `columns_iter`, `any_iter`) | the same, for `transform_iter` on a `pt.utils.peekable` iterator | |

- `extra_columns` lists the columns your code reads beyond the frame type's own, e.g. `["text"]`
  for a reranker or `["query_vec"]` for a dense retriever. Put every column you access there.
- Pass `context=self` so the error names the transformer.
- `inp` may also be a list of column names. That is what `transform_outputs` validates (section 6).
- `query_frame` requires only `qid`. Add `"query"` to `extra_columns` when you read the query text.
- Legacy code uses `pta.validate` from pyterrier-alpha. `pt.validate` is the same API in core.

## 3. Model-backed reranker (batched)

Load the model once in `__init__`. Score in batches in `transform`. Keep every input column and
replace `score`, then recompute ranks.

```python
import pyterrier as pt

class BatchedScorer(pt.Transformer):
    def __init__(self, model, batch_size=32, text_field="text"):
        self.model = model                 # loaded once, e.g. a CrossEncoder
        self.batch_size = batch_size
        self.text_field = text_field

    def transform(self, inp):
        pt.validate.result_frame(inp, extra_columns=["query", self.text_field], context=self)
        if len(inp) == 0:                  # inspection and Experiment validation send an empty frame
            return inp.assign(score=[], rank=[])
        scores = []
        for start in range(0, len(inp), self.batch_size):
            batch = inp.iloc[start:start + self.batch_size]
            pairs = list(zip(batch["query"], batch[self.text_field]))
            scores.extend(self.model.predict(pairs))
        return pt.model.add_ranks(inp.assign(score=scores))
```

- The empty-input branch must produce the same columns as the full path: here the input columns
  plus `score` and `rank` (`add_ranks` overwrites an existing `rank`).
- `pt.apply.doc_score(fn, batch_size=...)` does the same thing when you don't need a class.
- To keep the reranker's memory bounded, put a cut-off in front of it (`first_stage % k >> reranker`).

## 4. Query rewriter

Keep the original query in `query_0` with `pt.model.push_queries(..., keep_original=True)`. Without
`keep_original=True`, `push_queries` moves `query` to `query_0` and the frame no longer has a `query`
column.

```python
class AddTerm(pt.Transformer):
    def __init__(self, term):
        self.term = term

    def transform(self, inp):
        pt.validate.query_frame(inp, extra_columns=["query"], context=self)
        out = pt.model.push_queries(inp, keep_original=True)   # query_0 = previous query
        return out.assign(query=out["query"] + " " + self.term)

# Retrieve with the rewritten query, then restore the original for later stages
pipeline = AddTerm("synthesis") >> bm25 >> pt.rewrite.reset()
```

Restore the original query with `pt.rewrite.reset()` before a neural reranker, so the reranker sees
what the user typed. This class is empty-safe as written: string operations on an empty column
return an empty column.

## 5. One transformer, several modes

A transformer may accept more than one kind of input. pyterrier-dr's `BiEncoder` encodes queries
(Q in, `query_vec` out), encodes documents (D in, `doc_vec` out) or scores pairs (R in), and
`FlexIndex` retrieves from Q or rescores R. They choose by listing every accepted spec in a
`pt.validate.any` block and branching on `v.mode`, the first spec that matched:

```python
class DenseIndex(pt.Transformer):
    def transform(self, inp):
        with pt.validate.any(inp, context=self) as v:
            v.query_frame(extra_columns=["query_vec"], mode="retriever")
            v.result_frame(extra_columns=["query_vec"], mode="scorer")
        # raises InputValidationError, listing every spec, only if none matched
        if v.mode == "retriever":
            return self.retriever()(inp)
        return self.scorer()(inp)
```

- List specs from most to least specific. `v.mode` is the **first** match.
- `v.columns(includes=[...], excludes=[...], mode=...)` is the general form.
- `pt.inspect.transformer_inputs` returns one column list per spec, in order.
- Each branch still has to be empty-safe, so that `transformer_outputs` works for each mode.

## 6. Row-by-row transformer with `transform_iter`

A `transform_iter` transformer is not inspectable by itself: PyTerrier can't run it on an empty
frame and read the columns back. Validate the iterator for the input contract, and add
`transform_outputs` for the output contract. `transform_outputs` must validate the same way
`transform_iter` does, and return exactly the columns `transform_iter` yields.

```python
class UpperCaseQuery(pt.Transformer):
    def transform_iter(self, inp):
        inp = pt.utils.peekable(inp)       # re-assign, or you lose the first row
        pt.validate.query_iter(inp, extra_columns=["query"], context=self)
        for row in inp:                    # each row is a dict
            yield {**row, "query": row["query"].upper()}

    def transform_outputs(self, input_columns):
        pt.validate.query_frame(input_columns, extra_columns=["query"], context=self)
        return input_columns               # same columns in and out
```

- A transformer that adds columns returns `input_columns + ["docno", "score", "rank"]` or similar.
- `transform_inputs(self)` returning `["qid", "query"]` (or a list of such lists) is optional here:
  the `query_iter` call already lets `transformer_inputs` infer it. Add it when the
  validation happens too late to be seen on an empty input.
- The docs' advice is to implement `transform_outputs` **only** for `transform_iter` or when running
  the transformer on an empty frame is expensive. Otherwise it is one more thing that can drift
  from `transform`.

## 7. `pt.apply` transformers and inspection

`pt.apply.query`, `doc_score`, `doc_features` and `pt.apply.<column>` validate their default inputs
and are inspectable. The generic ones can't know what your function does:

| Apply form | Inspectable? | Make it so |
|---|---|---|
| `pt.apply.generic(fn)`, `pt.apply.by_query(fn)` | yes, if `fn` handles an empty frame and returns the right columns | pass `required_columns=[...]` for a clear input error |
| `pt.apply.generic(fn, iter=True)`, `pt.apply.by_query(fn, iter=True)` | no | pass `transform_outputs=lambda cols: cols + ["new_col"]` |
| any apply whose `fn` breaks on an empty frame | no | pass `transform_outputs=` |

```python
drop_low = pt.apply.generic(
    lambda rows: (r for r in rows if r["score"] > 0),
    iter=True,
    required_columns=["qid", "docno", "score"],
    transform_outputs=lambda cols: cols,
    label="drop_low",
)
```

## 8. Checking the contract

Do this once for every new transformer, before putting it in an experiment:

```python
t = BatchedScorer(model)
print(pt.inspect.transformer_inputs(t))                                     # [['qid', 'docno', 'query', 'text']]
print(pt.inspect.transformer_outputs(t, ["qid", "query", "docno", "text"]))  # [..., 'score', 'rank']

pipeline = bm25 % 100 >> index.text_loader() >> t
print(pt.inspect.transformer_outputs(pipeline, ["qid", "query"]))           # whole pipeline, no data needed

# Fail fast instead of a warning, before any system is run
pt.Experiment([pipeline], topics, qrels, ["ndcg_cut_10"], validate="error")
```

| Result | Meaning | Fix |
|---|---|---|
| `InputValidationError … missing_columns=[...]` | the pipeline stages don't fit | insert the stage that produces the missing column (`text_loader()`, a query encoder, `pt.rewrite.reset()`) |
| `InspectError: Cannot determine inputs/outputs for …` | the transformer crashed on an empty frame, or validates after reading a column | move validation to the first line; add the empty-input path |
| `InspectError: This transformer is not inspectable` | `transform_iter` without `transform_outputs` | section 6 |
| output list is missing a column the next stage needs | the empty-input branch and the full path disagree | make both paths return the same columns |

## 9. Checklist

- [ ] First statement of `transform` validates the input (`pt.validate.result_frame`, `query_frame`,
      `document_frame`, `columns`, or an `any` block for several modes), with every column the code reads in
      `extra_columns` and `context=self`.
- [ ] An empty input returns an empty frame with exactly the output columns, without running the model.
- [ ] `transform_iter` transformers validate a `pt.utils.peekable` iterator and implement `transform_outputs`.
- [ ] `pt.inspect.transformer_inputs(t)` and `pt.inspect.transformer_outputs(t, cols)` return what you expect.
- [ ] Returns `qid` and `docno` as strings, and keeps the input columns later stages may need.
- [ ] Recomputes ranks with `pt.model.add_ranks` after changing scores.
- [ ] Loads models once and scores in batches.
- [ ] Imports nothing from legacy APIs (`TransformerBase`, `pta.DataFrameBuilder`, `pta.validate`).
