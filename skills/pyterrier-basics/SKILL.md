---
name: pyterrier-basics
description: >-
  Core PyTerrier knowledge that every other pyterrier-* skill builds on: how
  PyTerrier works (transformers over query and result DataFrames), installing
  pyterrier and Java, the data model (qid, query, docno, score, rank), pipeline
  operators (>>, %, +, **, |, &), pt.apply, writing a custom pt.Transformer,
  replacing legacy APIs (pt.init, BatchRetrieve) with their 1.x forms, and the
  checklist for building a retrieval pipeline with a reranker. Use whenever writing,
  fixing or reviewing any PyTerrier code, even when another pyterrier skill also applies.
compatibility: pyterrier>=1.1 (checked with 1.1.2, Java 11)
---

# PyTerrier basics

Docs: [Installation](https://pyterrier.readthedocs.io/en/latest/installation.html) ·
[Data model](https://pyterrier.readthedocs.io/en/latest/datamodel.html) ·
[Transformers](https://pyterrier.readthedocs.io/en/latest/transformer.html) ·
[Operators](https://pyterrier.readthedocs.io/en/latest/operators.html) ·
[pt.apply](https://pyterrier.readthedocs.io/en/latest/apply.html) ·
[Java troubleshooting](https://pyterrier.readthedocs.io/en/latest/troubleshooting/java.html)

This is the base skill. The other pyterrier skills assume you know what is here.

## How PyTerrier works

- Everything is a `pt.Transformer`: it takes a DataFrame and returns a DataFrame.
- Retrievers turn queries into ranked results. Rerankers rescore results. Query rewriters change
  the `query` column. Indexers consume documents.
- Transformers are combined with operators into pipelines. A pipeline is itself a transformer, so
  it can be called with `.search("text")`, `.transform(df)`, or passed to `pt.Experiment`.
- Most work is choosing transformers for each stage. Each stage has its own skill:

| Stage | Skill |
|---|---|
| Datasets, evaluation, significance, tuning | `pyterrier-experiments` |
| Prebuilt indexes and caches on Hugging Face | `pyterrier-huggingface` |
| First-stage retrieval (Terrier, PISA, SPLADE, dense, ColBERT) | `pyterrier-first-stage` |
| Query rewriting, PRF, document expansion | `pyterrier-expansion` |
| Reranking (cross-encoders, T5, LLMs, learning to rank) | `pyterrier-reranking` |
| Retrieval-augmented generation | `pyterrier-rag` |
| Caching expensive stages | `pyterrier-caching` |

## Install

```bash
pip install 'pyterrier[all]'
```

- Install the package `pyterrier`. `python-terrier` is the old name. Don't install both.
- Needs Python 3.9 or newer. Terrier indexes and retrievers also need Java 11 or newer.
  Other backends, such as PISA or dense indexes, don't need Java.
- Java starts by itself the first time something needs it. Don't call `pt.init()`.
- Any Java setting must be made **before** Java starts, so put it straight after the import:

```python
import pyterrier as pt
pt.java.set_memory_limit(8.0)          # GB of JVM heap
# pt.java.set_java_home("/usr/lib/jvm/java-17-openjdk")
# pt.terrier.set_version("5.11")
```

## Quick start

```python run
import pyterrier as pt

index = pt.terrier.TerrierIndex.example()   # small prebuilt index, downloaded once
bm25 = index.bm25()
results = bm25.search("chemical reactions")
assert {"qid", "query", "docno", "score", "rank"} <= set(results.columns)
print(results.head())
```

## Data model

| Frame | Columns | Produced by |
|---|---|---|
| Q (queries, also called topics) | `qid`, `query` | `pt.new.queries([...])`, `dataset.get_topics()` |
| D (documents) | `docno`, plus text fields such as `text` | `dataset.get_corpus_iter()` (yields dicts) |
| R (results) | `qid`, `query`, `docno`, `score`, `rank` | retrievers and rerankers |
| qrels (judgements) | `qid`, `docno`, `label` | `dataset.get_qrels()` |

- `qid` and `docno` are **strings**. Convert ids you build yourself with `str(...)`.
- `rank` starts at 0, and the highest score in each query has rank 0.
- Terrier results also carry an integer `docid`, which is Terrier's internal id. Join and
  evaluate on `docno`.
- Results carry document text only when something adds it. Check with
  `"text" in results.columns`.
- A query rewriter keeps the previous query in `query_0`. `pt.rewrite.reset()` restores it.
- Extra columns pass through. Use `pt.inspect.transformer_outputs(t, ["qid", "query"])` to see what a
  transformer will output.

## Operators

| Operator | Meaning | Example |
|---|---|---|
| `a >> b` | then: feed a's output into b | `bm25 >> reranker` |
| `a % k` | keep the top k per query | `bm25 % 100` |
| `a + b` | add scores (linear combination) | `bm25 + 0.5 * dph` |
| `w * a` | multiply scores by a constant | `0.5 * dph` |
| `a ** b` | feature union, building a `features` column for learning to rank | `bm25 >> (tf ** pl2)` |
| `a \| b` | set union of documents (scores are dropped) | `bm25 \| dph` |
| `a & b` | set intersection of documents (scores are dropped) | `bm25 & dph` |
| `a ^ b` | concatenate b's results after a's | `bm25 % 10 ^ dph` |

Precedence is Python's: `**` > `*` `%` > `+` > `>>` > `&` > `^` > `|`. Don't rely on it.
Parenthesise every mixed expression, and always parenthesise `&`, `|`, `^` and `**`.

```python
# Incorrect: >> binds tighter than |, so this is (bm25 >> tf) | pl2
pipe = bm25 >> tf | pl2
# Correct: say what you mean
pipe = bm25 >> (tf | pl2)
```

For more worked precedence cases, read [references/operators.md](references/operators.md) whenever you
combine three or more transformers.

## pt.apply: small transformers from functions

| Need | Use |
|---|---|
| Rewrite each query | `pt.apply.query(lambda row: row["query"] + " extra")` |
| Rescore each document | `pt.apply.doc_score(fn)`, with fn(row) returning a float |
| Rescore in batches (neural models) | `pt.apply.doc_score(fn, batch_size=64)`, with fn(df) returning one score per row |
| Add a learning-to-rank feature | `pt.apply.doc_features(fn)`, with fn(row) returning a numpy array |
| Add or replace one column | `pt.apply.<column>(fn)`, e.g. `pt.apply.title_len(lambda row: len(row["title"]))` |
| Drop a column | `pt.apply.<column>(drop=True)` |
| Rename columns | `pt.apply.rename({"old": "new"})` |
| Anything else, per query | `pt.apply.by_query(fn)`, with fn(df_for_one_query) returning a df |
| Anything else, whole frame | `pt.apply.generic(fn)`, with fn(df) returning a df |

- Every `pt.apply` function takes `label="..."`, which names it in schematics and errors.
- `doc_score` recomputes `rank` for you.
- Use the batch form for anything model-based. Without `batch_size` the function is called once per row.

```python run
# Rerank the BM25 top 50 by document length (only to show the shape of a pipeline)
def length_score(df):
    return df["text"].str.len()

pipeline = (
    bm25 % 50
    >> index.text_loader()          # adds the `text` column that the scorer needs
    >> pt.apply.doc_score(length_score, batch_size=16, label="length")
)
reranked = pipeline.search("chemical reactions")
assert len(reranked) == 50 and reranked["rank"].min() == 0
```

## Writing a custom transformer

Use a `pt.Transformer` subclass when the component has state (a model or an index) or needs
configuration. Use `pt.apply` for one-liners. The template below is a reranker:

```python run
class LengthReranker(pt.Transformer):
    """Scores each retrieved document by its text length."""

    def transform(self, inp):
        pt.validate.result_frame(inp, extra_columns=["text"])   # clear error if text is missing
        out = pt.new.DataFrameBuilder(["qid", "query", "docno", "score"])
        for qid, group in inp.groupby("qid"):
            out.extend({
                "qid": qid,                          # a scalar repeats for every row
                "query": group["query"].iloc[0],
                "docno": group["docno"].tolist(),    # docnos, not scores
                "score": group["text"].str.len().tolist(),
            })
        return pt.model.add_ranks(out.to_df())

top = (bm25 % 10 >> index.text_loader() >> LengthReranker()).search("chemical reactions")
assert top.iloc[0]["rank"] == 0
```

- Validate inputs with `pt.validate.result_frame(...)` for results or `pt.validate.query_frame(...)` for queries.
- Build outputs with `pt.new.DataFrameBuilder` (in core since 1.1), and rank them with `pt.model.add_ranks`.
- Score documents in batches inside `transform`. Don't call a model once per row.

For model-backed transformers, query rewriters, `transform_iter`, and making a transformer work with
`pt.inspect`, read [references/custom-transformers.md](references/custom-transformers.md).

## Legacy → 1.x

Code from tutorials, papers and extension READMEs often predates 1.0. Translate it before you use it.

| Legacy | 1.x |
|---|---|
| `pt.init()`, `if not pt.started(): pt.init()` | delete it; Java starts by itself. For settings, use `pt.java.*` / `pt.terrier.*` (see Install) |
| `pt.init(mem=8000)` | `pt.java.set_memory_limit(8.0)` |
| `pt.BatchRetrieve(index, wmodel="BM25")`, `pt.TerrierRetrieve` | `pt.terrier.Retriever(index, wmodel="BM25")`, or `pt.terrier.TerrierIndex(path).bm25()` |
| `pt.FeaturesBatchRetrieve` | `pt.terrier.FeaturesRetriever` |
| `pt.BatchRetrieve.from_dataset(...)` | `pt.terrier.Retriever.from_dataset(...)` |
| `pyterrier.transformer.TransformerBase` | `pt.Transformer` |
| `pta.DataFrameBuilder`, `pta.validate` | `pt.new.DataFrameBuilder`, `pt.validate` |
| `~transformer` (caching operator) | pyterrier-caching (see the `pyterrier-caching` skill) |
| `pt.DFIndexer` | `pt.IterDictIndexer` or `pt.terrier.TerrierIndex(path).index(...)` |
| `import pyt_splade` | `import pyterrier_splade` |
| `pt.started()`, `pt.version()`, `pt.set_property()` | `pt.java.started()`, `pt.terrier.version()`, `pt.terrier.set_property()` |

## Workflow: creating an experimental pipeline

Copy this checklist and tick each item. Don't skip a gate.

```
- [ ] 1. Dataset and measures fixed from the task (pyterrier-experiments). Topics have a `query` column.
- [ ] 2. First-stage index: prebuilt artifact if one exists (pyterrier-huggingface), else build one (pyterrier-first-stage).
- [ ] 3. First-stage baseline evaluated.
         GATE: the score is in the expected range for this dataset. If not, stop and fix the index or parameters.
- [ ] 4. Optional expansion stage (pyterrier-expansion), evaluated against step 3.
- [ ] 5. Rerank depth k chosen and stated (`first_stage % k`). A `text` column is present before the reranker.
- [ ] 6. Reranker added (pyterrier-reranking). It runs on 2-3 queries with `.search()` before the full run.
- [ ] 7. All systems compared in one pt.Experiment, with the first stage as baseline and a multiple-testing correction.
- [ ] 8. Report: dataset id, measures, k, model names and versions, parameters, and the results table.
```

The pipeline shape this produces:

```python
pipeline = first_stage % k >> text_loader >> reranker
```

- Example code shows a technique, not a setup. Take the collection, topics, depth and measures from
  the user's task.
- Small collections (vaswani, `TerrierIndex.example()`) are only for checking that code runs.

### Comparing with published numbers

- Don't compare numbers from a toy collection with numbers from a paper.
- Default parameters differ between systems. For example, BM25 k1 and b differ between Terrier,
  Anserini and PISA. Pass parameters explicitly and check which values were actually used, from the
  installed package source, the index metadata or the paper. The `pyterrier-first-stage` skill lists
  indicative values.
- Report your numbers and the published numbers in separate columns, with the difference.
- Never change the evaluation to move a result. Record every assumption and every place where your
  setup differs from the publication.

## Don't

- Don't call `pt.init()` or check `pt.started()` before using PyTerrier.
- Don't use `pt.BatchRetrieve`, `pt.TerrierRetrieve` or `TransformerBase`. Use the 1.x names in the table above.
- Don't use the `~` caching operator. It was removed in 1.0.
- Don't create `pt.terrier.Retriever(index)` without `wmodel=`. The default is DPH, not BM25.
- Don't mix operators without parentheses.
- Don't give a reranker results that have no `text` column. Add `index.text_loader()` or `dataset.text_loader()` first.
- Don't use numeric `qid` or `docno` values. They must be strings.
- Don't score one row at a time with a neural model. Use `batch_size=` or batch inside `transform`.
- Don't copy snippets from the docs or READMEs unchecked. Some are broken or legacy. Run them first.
- Don't call `.parallel()` without `pyterrier-alpha[parallel]` installed. It is experimental.

## Reference files

| Read | When |
|---|---|
| [references/operators.md](references/operators.md) | combining three or more transformers, or debugging an unexpected pipeline structure |
| [references/custom-transformers.md](references/custom-transformers.md) | writing a transformer class: model-backed, query rewriting, iterator-based, or one that must work with `pt.inspect` |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Both 'pyterrier' and 'python-terrier' packages are installed` | an old `python-terrier` is installed next to `pyterrier` | `pip uninstall python-terrier` |
| Java not found, or a JVM version error on first retrieval | no Java 11+, or `JAVA_HOME` points to an old JVM | install Java 11+, or `pt.java.set_java_home(path)` straight after the import |
| `RuntimeError: You can only call <function set_memory_limit …> before either you start using java` | a Java setting was made after the first Terrier call | move `pt.java.*` / `pt.terrier.set_*` calls to straight after `import pyterrier` |
| `RuntimeError: You can only call <function legacy_init …>` | legacy `pt.init()` called after Java started | delete `pt.init()` (see Legacy → 1.x) |
| `DeprecationWarning: Call to deprecated class BatchRetrieve` | legacy code | see the Legacy → 1.x table |
| `KeyError: 'text'`, or `InputValidationError … missing_columns=['text']` | the reranker received results without document text | insert `index.text_loader()` or `dataset.text_loader()` before it |
| Pipeline results look wrong after combining with `\|`, `&` or `+` | operator precedence | parenthesise. Print the pipeline (`print(pipe)`) to see its structure |
| Results all come from one retriever after `\|` or `&` | set operators drop scores | rescore after the set operation, e.g. `(a \| b) >> scorer` |
| Out of memory in Java on a large index | default JVM heap too small | `pt.java.set_memory_limit(16.0)` before Java starts |
| `ImportError: pyterrier-alpha[parallel] must be installed` | `.parallel()` was called | `pip install 'pyterrier-alpha[parallel]'` |
