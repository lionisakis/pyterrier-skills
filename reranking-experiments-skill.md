---
name: reranking-experiments-skills
description: This skills shows how to conduct experiments that involve rerankers

---

## 1. Defining a reranker transformer

You can easily make a transformer by defining a function that takes the text of the query and the text of the document, and returns a score.

```python
import pyterrier as pt
def _scorer_fn(query : str, document : str) -> float:
    ...
    return score

transformer = pt.apply.doc_score(lambda row: _scorer_fn(row["query"], row["text"]))
```

For instance, if you are using the sentence transformers API:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
def _scorer_fn(query : str, document : str) -> float:
    return model.predict([(query, document)])[0]
```

Once you have a transformer that works, you can, if appropriate built that into a Transformer class:

```python
class MyReranker(pt.Transformer):

    def __init__(self, args):
        ...

    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        # validate that you have the correct inputs for columns. if you need extra columns, add them here.
        pt.validate.result_frame(inp, ["query", "docno", "text"])
        result = pta.DataFrameBuilder(['docno', 'score'])

        for qid, group in inp.groupby("qid"):
            query = group.iloc[0]["query"]
            docnos = group["scores"].tolist()
            scores = ... # calculate all scores; you may want to push apply batching.
            result.extend({
                'qid': qid, # automatically repeats to the length of this batch
                'query': query, # ditto
                'docno': docnos,
                'score': scores
            })
        return pt.model.add_ranks(result.to_df())
```

## 2. Identify a baseline ranker.

You should choose an appropriate baseline ranker, either sparse or dense.

Sparse:
 - from a Terrier index: `index.bm25()`

Dense:
 - from a Dense index: `model.query_encoder() >> index.np_retriever()` 


## 3. Adding text, if necessary

Some retrievers may not return the raw text of documents. You can tell this when using `.search(query : str)`, by seeing no "text" in the output columns.

```python
has_text = "text" in retriever.search("what are chemical reactions").columns
```

If this is the case, then `retriever >> fetch_text` where `fetch_text`  is defined as follows:
 - from a Terrier index that has the raw text in the index: `fetch_text = pt.text.get_text(terrier_index)`
 - from a dataset from ir_datasets: `fetch_text = pt.text.get_text(pt.get_dataset("irds:..."))`
 

## 4. Evaluation

Use PyTerrier's experiment for comparing the reranker with the baseline.

```python
from pyterrier.measures import *
pt.Experiment(
    [
        baseline,
        baseline >> reranker,
    ]
    topics,
    qrels,
    measures
)
```
