---
name: pyterrier-generative
description: >-
  Reranks PyTerrier results with LLM listwise and setwise rerankers from
  pyterrier-generative: RankZephyr, RankVicuna, RankGPT (OpenAI), LiT5, any
  Hugging Face, vLLM or OpenAI chat model through StandardRanker, and custom
  prompts through GenerativeRanker. Covers installing from GitHub, the
  pyterrier-rag conflict, choosing a backend, the ranking algorithms (sliding
  window, single window, top-down partitioning, setwise) with their window,
  stride and depth settings, caching and reporting. Use whenever an LLM
  reranker, listwise reranking, RankGPT, RankZephyr, RankVicuna, LiT5, setwise,
  TDPart or "rerank with an LLM / GPT" appears in a PyTerrier task. Not for
  cross-encoders, monoT5, duoT5 or monoELECTRA (see pyterrier-reranking), or
  for generating answers from retrieved passages (see pyterrier-rag).
compatibility: pyterrier>=1.1 (checked with 1.1.2; pyterrier-generative 0.1.5 from GitHub 391c748, pyterrier-rag remove-prompt branch f76c0d4, transformers 5.19, torch 2.14 CPU; CPU only, no vLLM)
---

# LLM rerankers: pyterrier-generative

Assumes the `pyterrier-basics` skill. Choosing a reranker at all, and the rerank depth, is in the
`pyterrier-reranking` skill.

Docs: [pyterrier-generative](https://github.com/Parry-Parry/pyterrier-generative) ·
[pyterrier-rag backends](https://github.com/terrierteam/pyterrier_rag) ·
[Neural rerankers](https://pyterrier.readthedocs.io/en/latest/neural.html) ·
[RankGPT paper](https://arxiv.org/abs/2304.09542) · [RankZephyr paper](https://arxiv.org/abs/2312.02724)

## Install

The PyPI release (0.1.3) doesn't import (`No module named 'pyterrier_rag.prompt.jinja'`). Install 0.1.5
from GitHub, in its own environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU machine; skip on GPU
pip install 'pyterrier[java]' ir-datasets git+https://github.com/Parry-Parry/pyterrier-generative.git
pip uninstall -y torchvision torchaudio                                 # CPU torch only: see Troubleshooting
python -c "import pyterrier_generative as g; print(g.__version__)"     # 0.1.5
```

- It installs pyterrier-rag from the `remove-prompt` branch, replacing the PyPI 0.3.0 that the
  `pyterrier-rag` skill targets. Both report version 0.3.0. Keep LLM reranking and RAG pipelines in
  separate environments.
- vLLM is not installed. Add `pip install vllm` on a GPU machine, or pass `backend="hf"` (below).
- LiT5 needs `pyterrier_t5.modeling_fid`, which no released pyterrier-t5 has. Treat LiT5 as unavailable
  while `python -c "import pyterrier_t5.modeling_fid"` fails.

## The pipeline

```python
import pyterrier as pt
from pyterrier_generative import RankZephyr, Algorithm

dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.terrier.TerrierIndex.example().bm25()      # a stand-in first stage; see pyterrier-retrieval
ranker = RankZephyr.v1(backend="vllm", window_size=20, stride=10)   # GPU
pipeline = bm25 % 100 >> dataset.text_loader() >> ranker
```

| | Columns |
|---|---|
| Input | `qid, query, docno, text`, plus `score` (algorithms order the input by it) |
| Output | `qid, query, docno, text, rank, score` only. Other input columns are dropped |

- `score` is `n, n-1, …, 1` per query, from the LLM's order. It isn't a relevance score: don't
  interpolate it with other scores. Rank-based fusion is fine (the `pyterrier-fusion` skill).
- If the model's answer can't be parsed, the documents it didn't name keep their input order. A weak
  model therefore looks like "no change", not like an error. Compare against the first stage.

## Ready-made rankers

| Ranker | Model | Default backend | Runs on |
|---|---|---|---|
| `RankZephyr.v1()` | `castorini/rank_zephyr_7b_v1_full` | `vllm` | GPU (~15 GB) |
| `RankVicuna.v1()` | `castorini/rank_vicuna_7b_v1` | `vllm` | GPU (~14 GB) |
| `RankGPT.gpt35()`, `.gpt35_16k()`, `.gpt4()`, `.gpt4_turbo()` | OpenAI models | `openai` | API key; costs money per window |
| `StandardRanker(model_id, backend=...)` | any chat model | `openai` for `gpt-*` ids, else `vllm` | depends on the model |
| `LiT5(model_path="castorini/LiT5-Distill-large")` | FiD T5 | own | unavailable (see Install) |

`StandardRanker` uses the RankGPT prompt with an empty system prompt; `RankZephyr` and `RankVicuna` add
the RankLLM system prompt, `RankGPT` the RankGPT one. Keyword arguments shared by all of them:

| Argument | Meaning |
|---|---|
| `backend` | `"vllm"`, `"hf"` or `"openai"` |
| `algorithm` | see Ranking algorithms |
| `window_size`, `stride` | documents per LLM call; step between windows |
| `buffer`, `cutoff`, `max_iters` | TDPart only |
| `k` | setwise only |
| `context_size`, `max_new_tokens` | the backend's input and output limits |
| `truncate_docs`, `max_prompt_length` | shorten documents when a window doesn't fit |
| `device` | Hugging Face backend only |
| `api_key` | OpenAI; otherwise `OPENAI_API_KEY` is read |
| `model_args`, `generation_args`, `backend_kwargs` | passed to the pyterrier-rag backend; decoding is greedy |

Defaults, and how they differ from `GenerativeRanker`'s, are in [references/api.md](references/api.md).

## Backends

```python
import os
from pyterrier_generative import StandardRanker, RankGPT, RankZephyr

small = StandardRanker("Qwen/Qwen2.5-0.5B-Instruct", backend="hf", device="cpu", window_size=5)
zephyr_hf = RankZephyr.v1(backend="hf", device="cuda")      # no vLLM: slower, same model
gpt = RankGPT.gpt4(api_key=os.environ["OPENAI_API_KEY"], backend_kwargs={"timeout": 60})
```

- Without `backend=`, any id that doesn't start with `gpt-` goes to vLLM, which fails on a machine
  without it. Pass `backend="hf"` there.
- A 7B ranker on CPU is impractical. On CPU, use a small instruct model with `backend="hf"` to check that
  code runs, never to report effectiveness.

## Ranking algorithms

| `Algorithm.` | Parameters | What it does | LLM calls per query |
|---|---|---|---|
| `SLIDING_WINDOW` | `window_size`, `stride` | windows from the bottom of the list to the top, each moving the best documents up | about (depth − window) / stride + 1 |
| `SINGLE_WINDOW` | `window_size` | reranks only the top `window_size`; the rest keep their order | 1 |
| `TDPART` | `window_size`, `buffer`, `cutoff`, `max_iters` | top-down partitioning around a pivot to find the top `cutoff` | grows slowly with depth |
| `SETWISE` | `k` | heap sort with two-document LLM comparisons, stopping after the top `k` | about 2 × (depth + k × log2 depth) |

- Cost grows with depth: RankZephyr's paper setting is BM25 top-100, window 20, stride 10 (9 calls).
- `SINGLE_WINDOW` and `SETWISE` only order the top `window_size` / `k`. Measures deeper than that see
  the first stage's order.
- Windows of different queries are batched together; windows of one query run one after another.

## Caching and reporting

Listwise scores depend on the other candidates in the window, so a per-document `ScorerCache` is wrong
here. Cache the whole pipeline per query with `RetrieverCache` (the `pyterrier-caching` skill), and change
the path whenever the first stage, depth, model or algorithm settings change.

```python
from pyterrier_caching import RetrieverCache

cached = RetrieverCache("vaswani.bm25-100.rankzephyr-w20-s10.cache",
                        bm25 % 100 >> dataset.text_loader() >> ranker)
```

Report the model, backend, algorithm, `window_size`, `stride` (or `k`, `cutoff`), rerank depth and, for API
models, the date and the number of calls.

## Custom prompts and models

For your own Jinja prompt, a callable prompt or a backend object, read
[references/api.md](references/api.md).

## Don't

- Don't `pip install pyterrier-generative` from PyPI; 0.1.3 doesn't import. Install from GitHub.
- Don't install it into the environment of the `pyterrier-rag` skill. It replaces pyterrier-rag.
- Don't copy the README code as-is. It uses `pt.init()` and `pt.BatchRetrieve` (see `pyterrier-basics`).
- Don't rely on the default backend without vLLM installed. Pass `backend="hf"`.
- Don't run a 7B ranker on CPU, or report numbers from a tiny CPU model.
- Don't rerank more than about 100 documents with a sliding window without counting the LLM calls first.
- Don't cache an LLM reranker with `ScorerCache`. Use `RetrieverCache` on the whole pipeline.
- Don't combine the output `score` with other scores. It is a rank, not a relevance score.
- Don't compare runs with different window, stride or depth settings without saying so.
- Don't plan on LiT5 until `import pyterrier_t5.modeling_fid` works.

## Reference files

| Read | When |
|---|---|
| [references/api.md](references/api.md) | checking defaults, writing your own prompt, using `GenerativeRanker` with a backend object, or tuning truncation |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyterrier_rag.prompt.jinja'` | pyterrier-generative 0.1.3 from PyPI, or PyPI pyterrier-rag installed over the branch build | reinstall from GitHub in a fresh environment |
| `RuntimeError: operator torchvision::nms does not exist`, then `Could not import module 'BartForConditionalGeneration'` on import | a CUDA `torchvision` from PyPI next to CPU torch | `pip uninstall torchvision torchaudio`, or install them from the same CPU index as torch |
| `ImportError: Please install vllm to use VLLMBackend` when building a ranker | the default backend for non-`gpt-` models is vLLM | `backend="hf"`, or install vLLM on a GPU machine |
| `ModuleNotFoundError: No module named 'pyterrier_t5'` from `LiT5()` | LiT5 needs `pyterrier_t5.modeling_fid`, which isn't released | use another ranker |
| Validation error about `text` | no text loader in front | `dataset.text_loader() >> ranker` |
| Output order is the input order | the model's answer couldn't be parsed (too small a model, or a truncated answer) | check one window by hand (references/api.md); use a ranking-tuned model |
| Answers ignore some documents, or the order barely changes, on long documents | the Hugging Face backend truncates prompts longer than `context_size` (4096) at the end, cutting the instructions | `truncate_docs=True`, a smaller `window_size`, or passages |
| Very slow | depth / stride windows, each a full LLM call | smaller depth, `SINGLE_WINDOW` or `TDPART`, vLLM on a GPU, and `RetrieverCache` |
