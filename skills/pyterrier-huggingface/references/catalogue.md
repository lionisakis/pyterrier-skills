# Catalogue of known PyTerrier artifacts

Repository ids were checked against the Hub on 2026-10-08 (204 repositories tagged
`pyterrier-artifact`). Sizes are the compressed download measured on 2026-10-07, so the unpacked
index is larger. The Hub changes, so **search the tag before relying on this list**:
https://huggingface.co/datasets?other=pyterrier-artifact

Load any of these with `pt.Artifact.from_hf("<id>")`. Read the repository card before use.

## Contents
1. MS MARCO passage v1 (dev, TREC DL 2019/2020)
2. MS MARCO document v1 and MS MARCO v2 / v2.1
3. BEIR
4. TREC-COVID / CORD-19
5. RAG Wikipedia
6. SARA (sensitivity-aware search)
7. Vaswani and samples (testing only)
8. Other collections
9. `pyterrier/from-dataset` branches
10. Repositories to skip

## 1. MS MARCO passage v1 (8.8M passages)

| Id | Kind | Notes | Size |
|---|---|---|---|
| `pyterrier/msmarco-passage.terrier` | Terrier | BM25 | 3.2 GB |
| `macavaney/msmarco-passage.terrier` | Terrier | | 3.2 GB |
| `macavaney/msmarco-passage.pisa` | PISA | porter2 stemming. Card reports DL'19 BM25 nDCG@10 0.4989 | 8.1 GB on the Hub (holds `.tar` and `.tar.lz4`; one is downloaded) |
| `macavaney/msmarco-passage.anserini` | Anserini | Card reports DL'19 BM25 nDCG@10 0.5121 | 3.8 GB |
| `macavaney/msmarco-passage.tasb.flex` | dense | TAS-B (`pyterrier_dr.TasB`) | 27 GB |
| `ntonellotto/msmarco-passage.{tasb,tct-hnp,ance}.flex` | dense | TAS-B, TCT-ColBERT-HNP, ANCE | 27 GB each |
| `pyterrier/msmarco_psg_v1.colbertv2` | ColBERT | `colbert-ir/colbertv2.0`, PLAID | 24 GB |
| `jpq-repro/msmarco-passage-train__tct_colbert__faiss2opq__M96_nbits8__ps159744__neg200__ibn__lr__` | JPQ | TCT-ColBERT, compressed | 1.3 GB |
| `macavaney/msmarco-passage.splade-lg.cache` | indexer cache | SPLADE-large document vectors | 28 GB |
| `macavaney/msmarco-passage.monoelectra.cache` | scorer cache | monoELECTRA scores | 3.5 GB |
| `macavaney/msmarco-passage.corpusgraph.bm25.{16,128,1024}` | corpus graph | BM25 neighbours, k = 16/128/1024 | 0.8 / 6.2 / 45 GB |
| `macavaney/msmarco-passage.corpusgraph.bm25.128.laff`, `.corpusgraph.tcthnp.16` | corpus graph | | 6.7 / 0.7 GB |
| `pyterrier-quality/<model>.msmarco-passage.cache` | quality scores | models: qt5-tiny, qt5-small, qt5-base, rand, cdd, epic, gpt2-ppl, itn, t5-ppl, tasb-mag | under 40 MB each |

Indexes built for the judged TREC DL topics (by Jack McKechnie). The dense ones are full-corpus size:

| Id pattern | Kind |
|---|---|
| `JackMcKechnie/msmarco_passage_trec_dl_20{19,20}_judged.terrier` | Terrier (1 GB) |
| `JackMcKechnie/msmarco_passage_trec_dl_20{19,20}_judged_<model>.flex` | dense, 27 GB each. Models: `Shitao_RetroMAE_MSMARCO_finetune`, `castorini_tct_colbert_msmarco`, `sebastian_hofstaetter_distilbert_dot_tas_b_b256_msmarco`, `sentence_transformers_msmarco_roberta_base_ance_firstp`, and `intfloat_e5_base_v2` (2019 only) |
| `JackMcKechnie/msmarco_passage_trec_dl_2019_judged_splade_naver_splade_{cocondenser_selfdistil,v2_distil,v3}.terrier` | SPLADE impacts in Terrier (9–14 GB, 2019 only) |

## 2. MS MARCO document v1 and MS MARCO v2 / v2.1

- MS MARCO document v1 is only available through `pt.Artifact.from_dataset("msmarco_document", ...)`
  (see section 9).

| Id | Kind | Notes | Size |
|---|---|---|---|
| `macavaney/msmarco-passage-v2.pisa` | PISA | 138M passages | 35 GB |
| `macavaney/msmarco-passage-v2-dedup.pisa`, `.terrier` | PISA, Terrier | deduplicated v2 | 36 / 15 GB |
| `pyterrier/msmarco-passage-v2.tasb.flex` | dense | TAS-B | **421 GB** |
| `pyterrier/msmarco-passage-v2.splade-lg.pisa` | PISA | SPLADE-large | **270 GB** |
| `pyterrier/msmarco-passage-v2-dedup.tcthnp.flex` | dense | TCT-ColBERT-HNP | **376 GB** |
| `namawho/msmarco-segment-v2.1.pisa` | PISA | TREC RAG segments | **97 GB** |

## 3. BEIR

Thirteen corpora under `pyterrier/`: `arguana`, `climate-fever`, `dbpedia-entity`, `fever`, `fiqa`,
`hotpotqa`, `nfcorpus`, `nq`, `quora`, `scidocs`, `scifact`, `trec-covid`, `webis-touche2020`.
Each has five artifacts:

| Id | Kind | Query with |
|---|---|---|
| `pyterrier/<corpus>.terrier` | Terrier | `.bm25()` |
| `pyterrier/<corpus>.pisa` | PISA (porter2) | `.bm25()` |
| `pyterrier/<corpus>.retromae.flex` | dense, 768-d | `pyterrier_dr.RetroMAE.msmarco_distill()` as the encoder (check the card) |
| `pyterrier/<corpus>.splade-v1.cache`, `.splade-v3.cache` | indexer cache | SPLADE document vectors, to index with PISA or Terrier |

Collections: [PISA](https://huggingface.co/collections/pyterrier/beir-pisa-indexes-6704f1cb9b51f8eaee21bac2) ·
[Terrier](https://huggingface.co/collections/pyterrier/beir-terrier-indexes-670569c5b0ec83825723253d) ·
[RetroMAE](https://huggingface.co/collections/pyterrier/beir-retromae-indexes-670bb95e54fda45dfa5d3f0e)

- Small (under 0.2 GB each): arguana, nfcorpus, scifact, scidocs, fiqa.
- Medium (under 1.2 GB each): quora, trec-covid, webis-touche2020.
- Large: nq (Terrier 1.4 GB, dense 8.2 GB), hotpotqa (1.9 / 16 GB), dbpedia-entity (1.9 / 14 GB), fever
  and climate-fever (they share a corpus: 3.0 / 17 GB).
- There are no artifacts here for BEIR's msmarco or cqadupstack.

## 4. TREC-COVID / CORD-19

| Id | Kind | Size |
|---|---|---|
| `macavaney/cord19.terrier` | Terrier | 47 MB |
| `macavaney/cord19.pisa` | PISA | 0.5 GB |
| `macavaney/cord19.tct-hnp.flex` | dense, TCT-ColBERT-HNP | 0.6 GB |
| `macavaney/cord19.trec-covid.monoelectra.cache` | scorer cache | 79 MB |
| `macavaney/cord19.corpusgraph.bm25.16` | corpus graph | 20 MB |

The BEIR version is `pyterrier/trec-covid.*` (section 3).

## 5. RAG Wikipedia

| Id | Kind | Notes | Size |
|---|---|---|---|
| `pyterrier/ragwiki-terrier` | Terrier | Wikipedia 2018 passages (`rag:nq_wiki`), stores title and text | 13 GB |
| `pyterrier/ragwiki-e5.flex` | dense | E5, 21M passages | 65 GB |
| `pyterrier/raghpq-terrier` | Terrier | HotpotQA Wikipedia (`rag:hotpotqa_wiki`) | 1.9 GB |
| `astappiev/ragwiki-corpusgraph` | corpus graph | BM25, k = 16 | 2.0 GB |
| `jpq-repro/nq-train__tct_colbert__faiss2opq__M96_nbits8__ps159744__neg200__ibn__lr__` | JPQ | | 2.5 GB |

## 6. SARA (sensitivity-aware search)

These indexes come from "A Sensitivity-Aware Test Collection for Search Among Personal Information"
(McKechnie, McDonald and Macdonald, SIGIR 2026, [arXiv 2606.27559](https://arxiv.org/abs/2606.27559)).
The collection is built from Enron emails, and the topics and qrels are distributed through ir_datasets.

| Id pattern | Kind |
|---|---|
| `JackMcKechnie/sara.terrier`, `sara_large.terrier`, `sara_small.terrier`, `sara_small_full_doc.terrier`, `sara_small_full_doc_v2.terrier` | Terrier (under 40 MB) |
| `JackMcKechnie/{sara,sara_large}_<model>.flex` | dense (about 0.4 GB). Models as in section 1, including `intfloat_e5_base_v2` |
| `JackMcKechnie/sara_small_<model>.flex` | dense (about 43 MB), without E5 |
| `JackMcKechnie/{sara,sara_large}_splade_naver_splade_{cocondenser_selfdistil,v2_distil,v3}.terrier` | SPLADE impacts in Terrier |

## 7. Vaswani and samples (testing only)

`pyterrier/vaswani.terrier` (1 MB), `pyterrier/vaswani.colbertv2`, `macavaney/vaswani.anserini`,
`macavaney/vaswani.splade.cache`, `pyterrier/sample.terrier` (what `TerrierIndex.example()` loads), and
`pyterrier/sample.flex`.

## 8. Other collections

| Id | Collection |
|---|---|
| `pyterrier-tutorial/bright.sustainable.*` | BRIGHT sustainable-living subset (Terrier, PISA, Anserini, SPLADE, dense, ColBERT) |
| `XiaoW0728/terrier_index_bright` | BRIGHT (Terrier) |
| `jueri/longeval-2026-snapshot-{1,2,3}-index` | LongEval 2026 (Terrier) |
| `theresiavr/legalpincite_doc_{dev,test}_{case,par}.terrier` | legal pin-cite retrieval |
| `pyterrier-quality/*.cw22b-en.cache`, `macavaney/*cw22*.cache` | ClueWeb22 quality scores (ClueWeb22 licence needed to use) |
| `arsalanh/wiki_index` | Wikipedia, 4096-d dense, 105 GB |

## 9. `pyterrier/from-dataset` branches

This is the old PyTerrier data repository. Load it with
`pt.Artifact.from_dataset("<dataset>", "<variant>")`. Variants include:

- `msmarco_passage`:
  - Terrier: `terrier_stemmed`, `terrier_unstemmed`, `terrier_stemmed_text`, `terrier_unstemmed_text`,
    `terrier_stemmed_docT5query`, `terrier_stemmed_deepct`.
  - PISA: `pisa_porter2`, `pisa_unstemmed`, `pisa_d2q_n40`, `pisa_d2q_n80`, and the Doc2Query−−
    variants `pisa_d2qmm_n{40,80}_{tct_p50,monot5_p60,electra_p70}`.
  - Dense: `ance`, `numpy_tcthnp`.
- `msmarco_document`: `terrier_stemmed`, `terrier_unstemmed`, `terrier_stemmed_text`,
  `terrier_unstemmed_text`, `terrier_stemmed_docT5query`.
- `msmarcov2_passage`, `msmarcov2_document`: Terrier variants (15–62 GB).
- `trec-covid`: `terrier_stemmed`, `terrier_unstemmed` (also `_text` and `_positions` versions).
- `vaswani`: `terrier_stemmed`, `terrier_unstemmed`, `terrier_stemmed_text`.

`_text` variants store the document text, so `index.text_loader()` works.

## 10. Repositories to skip

`macavaney/my-index.*`, `macavaney/my_index.*`, `macavaney/my-index2.pisa` and `deleteme*` are tests.
Some `jueri/*-index-dense` repositories are tagged as Terrier indexes. Read their cards before use.
