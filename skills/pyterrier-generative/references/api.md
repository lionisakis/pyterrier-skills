# pyterrier-generative API: defaults, prompts and backends

Read from the source of pyterrier-generative 0.1.5 (GitHub commit 391c748) with pyterrier-rag from the
`remove-prompt` branch (f76c0d4). Defaults change between commits: check them in your install with
`inspect.signature(StandardRanker.__init__)` and `inspect.signature(GenerativeRanker.__init__)`.

## Contents
1. Defaults
2. A custom prompt
3. GenerativeRanker with a backend object
4. Checking one window by hand
5. What was checked

## 1. Defaults

| Argument | `StandardRanker`, `RankZephyr`, `RankVicuna`, `RankGPT` | `GenerativeRanker` |
|---|---|---|
| `prompt` | `RANKPROMPT` (RankGPT wording) | required |
| `system_prompt` | `""`; RankLLM prompt for Zephyr/Vicuna; RankGPT prompt for RankGPT | `""` |
| `algorithm` | `Algorithm.SLIDING_WINDOW` | same |
| `window_size`, `stride` | 20, 10 | same |
| `buffer`, `cutoff`, `k` | 20, 10, 10 | same |
| `max_iters` | **100** | **10** |
| `truncate_docs` | **False** | **True** |
| `max_prompt_length` | None (the backend's `max_input_length`) | same |
| `truncate_tokens_per_iter` | 50 | same |
| `context_size` | 4096 (becomes the backend's `max_input_length`) | n/a: set on the backend |
| `max_new_tokens` | 100 | n/a: set on the backend |
| `backend` | `openai` if `model_id` starts with `gpt-`, else `vllm` | n/a: pass `model=` |
| decoding | greedy (`do_sample=False` for HF, `temperature=0.0` for vLLM/OpenAI), overridable via `generation_args` | the backend's |

Each LLM call asks for just enough new tokens to write `[1] > [2] > … > [w]` for its window, capped by
`max_new_tokens`.

## 2. A custom prompt

A string prompt is a Jinja template. It is rendered with these variables:

| Variable | Value |
|---|---|
| `query` | the query text |
| `num` | number of documents in the window |
| `passages` | list of document texts, in window order |
| `docs` | list of `(i, row)` pairs with `row.text` |

`[N]` inside document text is rewritten to `(N)` first, so it can't be mistaken for an identifier.

```python
from pyterrier_generative import StandardRanker, Algorithm

PROMPT = """Rank these {{ num }} abstracts by how well they answer: {{ query }}

{% for p in passages %}[{{ loop.index }}] {{ p }}
{% endfor %}
Answer only with identifiers, most relevant first, like [2] > [1]."""

ranker = StandardRanker("Qwen/Qwen2.5-7B-Instruct", backend="vllm", prompt=PROMPT,
                        system_prompt="You rank scientific abstracts.", window_size=10, stride=5)
```

A callable prompt receives `query`, `num` and `passages` as keyword arguments and returns a string or a
list of chat messages (`[{"role": ..., "content": ...}]`).

The answer is parsed by taking every number in it, in order, as a 1-based window position. Unknown or
repeated numbers are skipped and the documents never named keep their input order. Keep the
`[2] > [1]` instruction or something equivalent.

## 3. GenerativeRanker with a backend object

Build the backend with pyterrier-rag, then pass it as `model`:

```python
from pyterrier_rag.backend import HuggingFaceBackend
from pyterrier_generative import GenerativeRanker, Algorithm
from pyterrier_generative.prompts import RANKPROMPT, RANKGPT_SYSTEM_PROMPT

backend = HuggingFaceBackend(model_id="Qwen/Qwen2.5-0.5B-Instruct", max_input_length=4096,
                             max_new_tokens=100, generation_args={"do_sample": False}, device="cpu")
ranker = GenerativeRanker(model=backend, prompt=RANKPROMPT, system_prompt=RANKGPT_SYSTEM_PROMPT,
                          algorithm=Algorithm.SINGLE_WINDOW, window_size=10)
```

`VLLMBackend` and `OpenAIBackend` take the same `model_id`, `max_input_length`, `max_new_tokens` and
`generation_args`; `OpenAIBackend` also takes `api_key`, `base_url` (any OpenAI-compatible server),
`timeout` and `parallel`. Built directly, their `max_input_length` defaults to **512** tokens and
`max_new_tokens` to 32 (the Hugging Face backend uses the model's limit). A window of 20 passages doesn't
fit in 512: always set `max_input_length`.

## 4. Checking one window by hand

When the output order equals the input order, look at what the model actually wrote:

```python
docs = (bm25 % 5 >> dataset.text_loader())(topics.head(1))
prompt = ranker._build_prompt_for_texts(docs["text"].tolist(), docs["query"].iloc[0])
print(ranker.model.generate([prompt])[0].text)          # e.g. "[3] > [1] > [2] > [5] > [4]"
```

These are private methods of 0.1.5. If they have moved, render the template yourself with the variables
in section 2.

## 5. What was checked

On 2026-10-13, CPU only (no vLLM, no GPU), in a fresh environment built with the install lines in
SKILL.md:

- `import pyterrier_generative` succeeds after removing the PyPI `torchvision`; it fails with 0.1.3 from
  PyPI and with a CUDA torchvision next to CPU torch.
- `RankZephyr.v1()` without vLLM: `ImportError: Please install vllm to use VLLMBackend`.
- `LiT5()`: `ModuleNotFoundError: No module named 'pyterrier_t5'`.
- The signatures and defaults above, from the installed source.
- `StandardRanker("Qwen/Qwen2.5-0.5B-Instruct", backend="hf", device="cpu",
  algorithm=Algorithm.SINGLE_WINDOW, window_size=3)` in `bm25 % 3 >> dataset.text_loader() >> ranker` on
  one vaswani topic ran end to end and returned `qid, query, docno, text, rank, score` with scores
  `[3, 2, 1]`. The 0.5B model kept BM25's order. The machine was capped at 0.2 CPU core, so this took
  20 minutes, and the sliding-window and `RetrieverCache` checks did not finish.

The 7B rankers, RankGPT, TDPart and setwise were not run. Test them on two or three queries before a
full run.
