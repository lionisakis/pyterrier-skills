# PLAID-PRF (ColBERT v2 pseudo-relevance feedback)

Read from pyterrier_colbert2 at commit `4947df6` (`pyterrier_colbert2/ranking/_prf.py`), and run on CPU
over a 200-document index. See the `pyterrier-first-stage` skill for installing ColBERT v2.

## Usage

```python
import pyterrier as pt
import pyterrier_colbert2

index = pt.Artifact.from_hf(
    "pyterrier/msmarco_psg_v1.colbertv2",
    plaid_mode=True, ncells=4, centroid_score_threshold=0.4, ndocs=4096,
)
baseline = index.end_to_end()
prf = index.plaid_prf_end_to_end(top_psg=3, top_exp=14, beta=0.7)
```

`plaid_prf_end_to_end(k=1000, **kwargs)` is `plaid_prf(index, **kwargs) >> index.end_to_end(k=k, query_encoded=True)`.

## Parameters of `plaid_prf`

| Parameter | Default | Meaning |
|---|---|---|
| `top_psg` | 5 | feedback passages taken from the first retrieval |
| `top_exp` | 16 | expansion embeddings added to the query |
| `beta` | 0.7 | weight of the expansion embeddings |
| `lambda_div` | 0.3 | diversity weight in the MMR selection of expansion embeddings |
| `mmr_selection` | True | choose expansion embeddings by MMR |
| `dedup_same_wp` | True | drop repeated word pieces |
| `weighting` | `"tf-idf"` | `"tf-idf"`, `"rm1"`, `"rm3"`, `"bo1"`, or `"dfr"`. **Not `"dfr_rsj"`**, which always fails although the type hint lists it |
| `rm3_lambda` | 0.5 | mixing weight when `weighting="rm3"` |
| `temperature` | 1.0 | softmax temperature for `"rm1"` |
| `output_exptok` | False | also output the expansion tokens as text; needs `dataset=` and is slow |

The README's example uses `top_psg=3, top_exp=14, beta=0.7`.

## Behaviour to know

- The expanded query is produced as a `query_vec` column, which `end_to_end(query_encoded=True)`
  reads. That's why `index.query_encoder()` (which writes `query_embs`) can't be chained into
  `end_to_end(query_encoded=True)`.
- The first call computes corpus statistics and writes `stats.json`, `idf_map.json`, `df_map.json` and
  `cf_map.json` into the index folder. The folder must be writable, and on a large index this first
  call takes a long time.
- Queries are processed one at a time.
- Compare against `index.end_to_end()` with the same PLAID settings in one `pt.Experiment` (see the
  `pyterrier-experiments` skill).
