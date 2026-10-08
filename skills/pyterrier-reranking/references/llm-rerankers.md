# LLM rerankers with pyterrier-generative

Read from the source of pyterrier-generative 0.1.3 (PyPI), installed with PyTerrier 1.1.2 and
pyterrier-rag 0.3.0.

> **On 2026-10-08, `import pyterrier_generative` fails** with
> `ModuleNotFoundError: No module named 'pyterrier_rag.prompt.jinja'`. pyterrier-generative 0.1.3 needs
> a module that neither pyterrier-rag 0.3.0 (PyPI) nor the pyterrier_rag GitHub master contains.
> Before planning work around these rerankers, check whether `import pyterrier_generative` succeeds in
> the user's environment. If it fails, tell the user, and use a cross-encoder or monoT5 instead (see
> SKILL.md).

The API below is from the source, so check it again once the package imports. Section 5 records
what was checked.

## Contents
1. Install
2. Ready-made rankers
3. Ranking algorithms
4. A custom prompt or model
5. What was checked, and known problems

## 1. Install

```bash
pip install pyterrier-generative     # Python 3.10+; also installs pyterrier-rag
```

- It brings in `pyterrier-rag`, whose backends (Hugging Face, vLLM, OpenAI) run the models.
- The README uses legacy `pt.init()` and `pt.BatchRetrieve`. Translate them (see the
  `pyterrier-basics` skill).

## 2. Ready-made rankers

```python
from pyterrier_generative import RankZephyr, RankVicuna, RankGPT, LiT5, Algorithm

zephyr = RankZephyr.v1(window_size=20)                 # castorini/rank_zephyr_7b_v1_full
gpt = RankGPT.gpt35(api_key="...")                     # gpt-3.5-turbo; also gpt35_16k, gpt4, gpt4_turbo
pipeline = first_stage % 100 >> dataset.text_loader() >> zephyr
```

| Class | Model | Runs on |
|---|---|---|
| `RankZephyr.v1()` | `castorini/rank_zephyr_7b_v1_full` | GPU (Hugging Face or vLLM backend) |
| `RankVicuna.v1()` | `castorini/rank_vicuna_7b_v1` | GPU |
| `RankGPT.gpt35()`, `.gpt35_16k()`, `.gpt4()`, `.gpt4_turbo()` | OpenAI models | API key, costs money per call |
| `LiT5(model_path="castorini/LiT5-Distill-large")` | LiT5 (FiD T5) | GPU or CPU; see section 5 |

Common keyword arguments (from `_GenerativeRanker`): `window_size=20`, `stride=10`,
`algorithm=Algorithm.SLIDING_WINDOW`, `backend=None` (e.g. `"hf"`, `"vllm"`), `model_args`,
`generation_args`, `device`, `api_key`, `max_new_tokens=100`, `verbose`.

## 3. Ranking algorithms

`Algorithm.SLIDING_WINDOW` (default), `Algorithm.SINGLE_WINDOW`, `Algorithm.SETWISE`, `Algorithm.TDPART`.

- **Sliding window** reranks the list in overlapping windows of `window_size`, moving by `stride`.
  Its cost grows with depth ÷ stride. Keep the rerank depth small (20–100).
- **Single window** reranks the top `window_size` documents and leaves the rest unchanged.
- **Setwise** and **TDPart** are alternative strategies. Read the class docstrings for their parameters
  (`k`, `cutoff`, `buffer`, `max_iters`).

## 4. A custom prompt or model

```python
from pyterrier_generative import GenerativeRanker, Algorithm

ranker = GenerativeRanker(
    model=backend,                 # a pyterrier-rag backend (Hugging Face, vLLM or OpenAI)
    prompt=my_prompt,              # a Jinja template string, or a callable that builds the prompt
    algorithm=Algorithm.SLIDING_WINDOW,
    window_size=20, stride=10,
    truncate_docs=True, max_prompt_length=4096,   # shorten documents that don't fit
)
```

Backends are described in the `pyterrier-rag` skill.

## 5. What was checked, and known problems

- **The package doesn't import** with released dependencies (see the note at the top). Checked by
  installing it in a clean environment with CPU torch.
- **LiT5 doesn't import with the released packages.** `LiT5` imports `pyterrier_t5.modeling_fid`.
  pyterrier-t5 0.3.0 (PyPI) and the pyterrier_t5 GitHub master contain only `__init__.py`, so that
  module doesn't exist. pyterrier-generative also doesn't install pyterrier-t5 at all. Treat LiT5 as
  unavailable unless a newer pyterrier-t5 adds the module: `import pyterrier_t5.modeling_fid` tells you.
- No model in this package could be run, because of the import failure. The 7B and OpenAI models would
  also need a GPU or an API key. Check the package README of your installed
  version, and test on two or three queries with `.search()` before a full run.
- LLM rerankers are slow and, for API models, cost money per call. Cache their output (see the
  `pyterrier-caching` skill) and keep depth and window settings in the report.
