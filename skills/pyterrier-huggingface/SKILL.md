---
name: pyterrier-huggingface
description: >-
  Finds, loads and shares prebuilt PyTerrier artifacts (Terrier, PISA, dense
  FlexIndex, ColBERT and Anserini indexes, SPLADE and reranker score caches,
  corpus graphs) on the Hugging Face Hub with pt.Artifact.from_hf, to_hf,
  from_url and from_dataset. Covers discovering artifacts through the
  pyterrier-artifact tag and the BEIR collections, which package each artifact
  needs, download sizes, caching and how artifacts are stored. Use when a task
  needs an index for MS MARCO, TREC DL, BEIR, TREC-COVID, RAG Wikipedia or
  another standard collection, before building an index yourself, or when
  publishing an index. Not for building new indexes.
compatibility: pyterrier>=1.1 (checked with 1.1.2; Hub listings checked 2026-10-08)
---

# PyTerrier artifacts on Hugging Face

Assumes the `pyterrier-basics` skill.

Docs: [Artifacts](https://pyterrier.readthedocs.io/en/latest/artifacts/index.html) ·
[Artifact listing](https://pyterrier.readthedocs.io/en/latest/artifacts/listing.html) ·
[Artifact how-to (sharing and storage)](https://pyterrier.readthedocs.io/en/latest/artifacts/how-to.html) ·
Paper: [Artifact Sharing for Information Retrieval Research (SIGIR 2025)](https://arxiv.org/abs/2505.05434)

Building an index for a standard collection can take hours. Many are already on the Hub, so look
there first.

## Quick start

```python run
import pyterrier as pt

index = pt.Artifact.from_hf("pyterrier/vaswani.terrier")   # downloads once, then cached
results = index.bm25().search("chemical reactions")
assert results.iloc[0]["rank"] == 0
```

`pt.Artifact.from_hf` reads the artifact's metadata and returns the right class: here a
`pt.terrier.TerrierIndex`, so `.bm25()`, `.text_loader()` and the rest of its API are available.

## Finding an artifact

Search the Hub before naming a repository. Never invent a repository id from memory.

1. Every PyTerrier artifact is a Hugging Face **dataset** repository (not a model) tagged
   `pyterrier-artifact`.
   - Browse: https://huggingface.co/datasets?other=pyterrier-artifact
   - API: `https://huggingface.co/api/datasets?filter=pyterrier-artifact&limit=1000`
2. To narrow down by type, filter on the type tag, e.g. `filter=pyterrier-artifact.sparse_index.pisa`.
3. For BEIR, use the `pyterrier` collections:
   - [BEIR: PISA Indexes](https://huggingface.co/collections/pyterrier/beir-pisa-indexes-6704f1cb9b51f8eaee21bac2)
   - [BEIR: Terrier Indexes](https://huggingface.co/collections/pyterrier/beir-terrier-indexes-670569c5b0ec83825723253d)
   - [BEIR - RetroMAE Indexes](https://huggingface.co/collections/pyterrier/beir-retromae-indexes-670bb95e54fda45dfa5d3f0e)
4. Read the repository's card (README) for the corpus, model, settings and any benchmark scores.
   Check the file sizes on its "Files" tab before downloading.

```python
import requests
rows = requests.get(
    "https://huggingface.co/api/datasets",
    params={"filter": "pyterrier-artifact.sparse_index.pisa", "limit": 1000},
    timeout=60,
).json()
print(sorted(r["id"] for r in rows))
```

For known repositories by collection, with sizes and models, read
[references/catalogue.md](references/catalogue.md).

## Which package each artifact needs

The type tag `pyterrier-artifact.<type>.<format>` says which package loads it:

| Type tag | Loads as | Install | Query it with |
|---|---|---|---|
| `sparse_index.terrier` | `pt.terrier.TerrierIndex` | core (Java 11+) | `.bm25()`, `.dph()`, `.retriever(...)` |
| `sparse_index.pisa` | `pyterrier_pisa.PisaIndex` | `pyterrier-pisa` (Linux x86_64 only) | `.bm25()`, `.quantized()` |
| `sparse_index.anserini` | `AnseriniIndex` | `pyterrier-anserini` (Java) | `.bm25(...)` |
| `dense_index.flex` | `pyterrier_dr.FlexIndex` | `pyterrier-dr` | `encoder >> index.retriever()` with **the same encoder that built it** |
| `dense_index.jpq` | `JPQIndex` | `pyterrier-dr` | see the card |
| `dense_index.colbert` | `ColBERTv2Index` | `pyterrier_colbert2` (GPU) | `.end_to_end()` |
| `indexer_cache.lz4pickle` | `IndexerCache` | `pyterrier-caching` | iterate it to index, e.g. SPLADE vectors |
| `scorer_cache.hdf5` | `ScorerCache` | `pyterrier-caching` | in place of the reranker |
| `corpus_graph.np_topk` | corpus graph | `pyterrier-adaptive` | adaptive reranking |
| `quality_score_cache.numpy` | quality scores | `pyterrier-quality` | static pruning |

- A missing package raises `ValueError: No implementation found …`, followed by the install hint.
- A dense or learned-sparse index contains vectors only. Find the encoder on the card (e.g.
  RetroMAE for `*.retromae.flex`, E5 for `ragwiki-e5.flex`) and use exactly that model for queries.
- Most indexes don't store document text. Check `"text" in index.bm25().search("x").columns`, and
  if it's missing, load text from the dataset (`dataset.text_loader()`) before reranking.

## Loading options

| Need | Use |
|---|---|
| A Hub repository | `pt.Artifact.from_hf("owner/repo")` |
| A fixed version | `pt.Artifact.from_hf("owner/repo@<branch, tag or commit>")` or `branch=` |
| Verify the download | `expected_sha256="..."` |
| Constructor options (e.g. ColBERT PLAID settings) | extra keyword arguments, passed to the class |
| A specific class | `PisaIndex.from_hf(...)`, `pt.terrier.TerrierIndex.from_hf(...)` |
| An index from the old PyTerrier data repository | `pt.Artifact.from_dataset("vaswani", "terrier_stemmed")` (reads `pyterrier/from-dataset` branches) |
| Zenodo, or any URL | `pt.Artifact.from_url("zenodo:<id>")`, `from_url("hf:owner/repo")`, `from_url("https://…/artifact.tar.lz4")`. Core resolves `hf:` and `zenodo:`; extensions add other schemes |
| A folder on disk | `pt.Artifact.load(path)` |

## Downloads and the cache

- Downloads go to `$PYTERRIER_HOME/artifacts/<sha256 of the URL>/` (default `~/.pyterrier`), not to
  the Hugging Face cache. `HF_HOME` and `hf_transfer` have no effect.
- The archive is unpacked while it downloads. You need disk space for the **unpacked** index, which is
  larger than the size shown on the Hub.
- A finished download is never fetched again. `@main` keeps the first version you downloaded. To
  update it, pin a newer ref or delete that folder.
- Private repositories need `HF_TOKEN` set, or `huggingface-cli login`.
- **Ask before downloading more than about 5 GB.** Some MS MARCO v2 artifacts are hundreds of GB
  (TAS-B: 421 GB, SPLADE PISA: 270 GB).

## Publishing an artifact

```python
index = pt.terrier.TerrierIndex("./my-index.terrier")
index.to_hf("my-user/my-collection.terrier")   # needs HF_TOKEN with write access
```

This creates a dataset repository, uploads `artifact.tar.lz4` (split into 20 GB parts when larger)
with `artifact.tar.lz4.json`, and writes a card with the `pyterrier-artifact` tags. Fill in the
card's TODO sections. Publishing is outward-facing, so confirm the repository name and visibility
with the user first. For storage details, Zenodo, peer-to-peer transfer and manual hosting, read
[references/sharing.md](references/sharing.md).

## Don't

- Don't name a repository you haven't confirmed on the Hub.
- Don't look for PyTerrier artifacts among Hugging Face *models*. They are datasets.
- Don't download a large artifact without stating its size and getting agreement.
- Don't query a dense index with a different encoder from the one that built it.
- Don't assume an index stores document text.
- Don't compare a prebuilt index's scores with a paper without checking its stemming, stopwords and
  BM25 parameters on the card.
- Don't use the `macavaney/my-index*`, `macavaney/my_index*` or `deleteme*` repositories. They are tests.
- Don't use `pta.Artifact` from old cards. It is `pt.Artifact` in 1.x.

## Reference files

| Read | When |
|---|---|
| [references/catalogue.md](references/catalogue.md) | choosing an artifact for MS MARCO, TREC DL, BEIR, TREC-COVID, RAG Wikipedia, SARA or another collection |
| [references/sharing.md](references/sharing.md) | publishing an artifact, or loading from Zenodo, a URL or peer-to-peer |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: No implementation found … Do you need to pip install X?` | the package for that artifact type isn't installed | `pip install X` (see the table above) |
| HTTP 404 on download | wrong id, or a model repository id | search the `pyterrier-artifact` tag; artifacts are datasets |
| HTTP 401 or 403 | private or gated repository | set `HF_TOKEN` |
| Disk full during download | the unpacked size is larger than the Hub size | free space, or set `PYTERRIER_HOME` to a larger disk |
| Still the old version after the repository changed | the cache never refreshes | pin `@<commit>`, or delete `$PYTERRIER_HOME/artifacts/<hash>` |
| `KeyError: 'text'` in a reranker | the index stores no text | add `dataset.text_loader()` before the reranker |
| Dense retrieval returns poor results | wrong query encoder for the index | use the encoder named on the card |
| `pip install pyterrier-pisa` finds no matching distribution | it ships Linux x86_64 wheels only | use the `.terrier` version of the same collection, or run on Linux |
