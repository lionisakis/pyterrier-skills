# Neural rerankers

Checked with PyTerrier 1.1.2, pyterrier-t5 0.3.0, pyterrier-dr 0.8.0 and sentence-transformers 6.1,
on CPU, each on one query with three vaswani documents.

## Contents
1. sentence-transformers CrossEncoder
2. monoELECTRA
3. monoT5 and duoT5
4. ColBERT text scorer
5. Dense bi-encoder as a scorer
6. Speed

## 1. sentence-transformers CrossEncoder

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)

def score(df):
    return model.predict(list(zip(df["query"], df["text"])))

reranker = pt.apply.doc_score(score, batch_size=64, label="MiniLM")
```

- Any `CrossEncoder` checkpoint works. Choose one trained for retrieval, e.g. the
  `cross-encoder/ms-marco-*` family.
- `batch_size` in `doc_score` makes `score` receive a DataFrame, which is what the batched `predict`
  needs.
- For a reusable class, see the custom transformer template in the `pyterrier-basics` skill.

## 2. monoELECTRA

```python
import pyterrier_dr
reranker = pyterrier_dr.ElectraScorer(batch_size=16)
```

`ElectraScorer(model_name="crystina-z/monoELECTRA_LCE_nneg31", batch_size=16, text_field="text",
verbose=True, device=None)`.

## 3. monoT5 and duoT5

```python
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

mono = MonoT5ReRanker(batch_size=16)
duo = DuoT5ReRanker(batch_size=16)
pipeline = first_stage % 100 >> dataset.text_loader() >> mono % 10 >> duo
```

| Class | Defaults |
|---|---|
| `MonoT5ReRanker` | `tok_model="t5-base"`, `model="castorini/monot5-base-msmarco"`, `batch_size=4`, `text_field="text"`, `device=None` (CUDA if available), `verbose=True` (a progress bar per batch; pass `False` for clean logs) |
| `DuoT5ReRanker` | `tok_model="t5-base"`, `model="castorini/duot5-base-msmarco"`, `batch_size=4`, `text_field="text"`, `device=None`, `verbose=True`, `agg="sum"` |
| `mT5ReRanker` | multilingual monoT5: `tok_model` and `model` default to `unicamp-dl/mt5-base-mmarco-v2`, `batch_size=4`, `text_field="text"`, `verbose=True` |

- For other checkpoints (e.g. `castorini/monot5-3b-msmarco-10k`), pass `model=` and the matching
  `tok_model=`.
- monoT5 scores are log-probabilities of "true", so they are negative. Only their order matters.
- duoT5 scores each pair of documents, so its cost grows with the square of the depth. Use it on a
  small top-k after monoT5.
- Input longer than T5's 512 tokens is truncated. For long documents use passages (see SKILL.md,
  "Long documents").

## 4. ColBERT text scorer

See the `pyterrier-colbert` skill, which covers ColBERT as retriever, reranker and PRF.

## 5. Dense bi-encoder as a scorer

See the `pyterrier-dense` skill, which covers pyterrier-dr models as retriever, reranker and PRF.

## 6. Speed

- A GPU is effectively required for reranking full test collections. Every model here also runs
  on CPU, slowly.
- Throughput grows with `batch_size` until GPU memory runs out. Out-of-memory errors mean the batch
  size or `max_length` should be lowered.
- Rerank depth multiplies the cost. Measure with `pt.Experiment(..., eval_metrics=[..., "mrt"])`.
- Cache scores you will reuse (see the `pyterrier-caching` skill).
