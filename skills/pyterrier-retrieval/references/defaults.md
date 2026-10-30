# Default parameters: indicative values only

**Verify before relying on these.** Defaults change between systems, versions and even prebuilt
indexes. Check the values actually used with the commands in section 2, and the values a paper used
in the paper itself.

## 1. Indicative defaults

Checked on 2026-10-08 with PyTerrier 1.1.2 (Terrier 5.11) and pyterrier-pisa 0.4.7. The Anserini
values come from Anserini/Pyserini documentation and were not checked here.

| System | BM25 k1 | BM25 b | Stemmer | Stopwords |
|---|---|---|---|---|
| Terrier (`TerrierIndex.bm25()`) | 1.2 | 0.75 | Porter | Terrier English list |
| PISA (`PisaIndex.bm25()`) | 0.9 | 0.4 | Porter2 | Terrier English list |
| Anserini / Pyserini default | 0.9 | 0.4 | Porter | Lucene English list |
| Anserini values commonly used for MS MARCO passage | 0.82 | 0.68 | Porter | Lucene English list |

- `pt.terrier.Retriever(index)` without `wmodel=` uses **DPH**, not BM25.
- Other PISA defaults: `pl2(c=1.0)`, `qld(mu=1000.0)`.
- Prebuilt indexes may have been built with non-default stemming or stopwords. Read the card.

## 2. How to check

```python
import inspect
print(inspect.signature(index.bm25))        # Terrier or PISA defaults in the installed version
print(index.bm25(k1=0.9, b=0.4).controls)   # Terrier: {..., 'bm25.k_1': 0.9, 'bm25.b': 0.4}; print(retriever) shows only the model name
print(open(f"{index.path}/data.properties").read())   # Terrier: stemmer, stopwords, tokeniser
print(open(f"{index.path}/pt_meta.json").read())      # any artifact: type, format and other metadata
```

- For a published number, find the parameters in the paper or its code, and pass them explicitly.
- When comparing two systems, pass the same k1 and b to both, and say in the report that you did.
