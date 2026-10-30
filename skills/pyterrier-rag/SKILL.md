---
name: pyterrier-rag
description: >-
  Builds and evaluates retrieval-augmented generation (RAG) pipelines in
  PyTerrier with pyterrier-rag: retrieving passages, concatenating them into a
  context, generating answers with a reader over Hugging Face, vLLM or OpenAI
  backends, Fusion-in-Decoder readers (T5FiD, BARTFiD), agentic search
  (SearchR1, SearchO1, R1Searcher), QA datasets (Natural Questions, HotpotQA,
  TriviaQA and others via rag:), Wikipedia corpora and indexes, and answer
  measures (exact match, F1, ROUGE, BERTScore, LLM-as-judge). Use when asked to
  build, run or evaluate a RAG or open-domain question answering pipeline in
  PyTerrier. Not for retrieval-only evaluation or reranking with LLMs (see
  pyterrier-generative).
compatibility: pyterrier>=1.1 (checked with 1.1.2 and pyterrier-rag 0.3.0, transformers 5.19, CPU only)
---

# Retrieval-augmented generation

Assumes the `pyterrier-basics` skill.

Docs: [pyterrier-rag](https://pyterrier.readthedocs.io/en/latest/ext/pyterrier-rag/index.html) ·
[GitHub](https://github.com/terrierteam/pyterrier_rag)

```bash
pip install pyterrier-rag      # Python 3.10+
```

## The shape of a RAG pipeline

```
retriever % k  >>  text_loader  >>  Concatenator()  >>  Reader(backend)
```

| Stage | Output |
|---|---|
| retriever, text loader | results with `text` (see the `pyterrier-retrieval` skill) |
| `Concatenator()` | **one row per query**: `qid`, `query`, `qcontext` (the passages joined into one string) |
| `Reader(backend, prompt=...)` | `qid`, `query_0`, `prompt` (the text sent to the model), `qanswer` |

```python run slow
import pyterrier as pt
import pyterrier_rag as ptr
from pyterrier_rag.prompt import Concatenator, PromptTransformer
from pyterrier_rag.readers import Reader

index = pt.terrier.TerrierIndex.example()
dataset = pt.get_dataset("irds:vaswani")
retriever = index.bm25() % 3 >> dataset.text_loader()

backend = ptr.Seq2SeqLMBackend("google/flan-t5-small", max_new_tokens=16)
template = (
    "Use the context information to answer the Question: \n"
    " Context: {{ qcontext }} \n Question: {{ query }} \n Answer:"
)
reader = Reader(backend, prompt=PromptTransformer(instruction=template, raw_instruction=True))

rag = retriever >> Concatenator() >> reader
answer = rag.search("what are electroluminescent storage systems")
assert answer["qanswer"].iloc[0] != "### Assistant"
```

- `k`, the number of passages, is a quality and cost decision, as is the size of `qcontext`. Limit
  it with `Concatenator(max_elements=..., max_length=..., tokenizer=...)`.
- `search()` takes one query string. To answer a frame of topics without gold answers, call
  `answers = rag.transform(topics)`: one row per topic with `qid`, `query_0`, `prompt` and `qanswer`. With
  gold answers, use `pt.Experiment` ("Evaluating answers" below).
- Templates are Jinja: `{{ query }}` and `{{ qcontext }}`.

## Prompts: always check what the model receives

A string prompt given to `Reader` is wrapped in a FastChat conversation template, chosen from the
model name. For models FastChat doesn't know, e.g. FLAN-T5, it falls back to a one-shot template
that starts with an **unrelated example conversation** ("A chat between a curious human… Got any
creative ideas for a 10 year old's birthday?…"). The model then answers nonsense (`'### Assistant'`).

| Model type | Use |
|---|---|
| Plain seq2seq or base model (FLAN-T5, T5) | `PromptTransformer(instruction=template, raw_instruction=True)` |
| Chat model that FastChat knows (Llama, Mistral, Qwen, ...) | the default, or `PromptTransformer(instruction=template, model_name_or_path=..., system_message=...)` |
| OpenAI API | `OpenAIBackend`, which sends chat messages |

After building a pipeline, print `result["prompt"].iloc[0]` once and read it.

## Backends

| Backend | Model class | Notes |
|---|---|---|
| `ptr.HuggingFaceBackend(model_id, max_new_tokens=32, max_input_length=None, model_args={}, generation_args=None, device=None)` | `AutoModelForCausalLM` | decoder-only models; prompts are truncated to `max_input_length` (default: the model's maximum) |
| `ptr.Seq2SeqLMBackend(model_id, ...)` | `AutoModelForSeq2SeqLM` | T5, FLAN-T5, BART |
| `ptr.VLLMBackend(model_id, ...)` | vLLM | fast batched GPU generation; needs `vllm` |
| `ptr.OpenAIBackend(model_id, api_key=..., base_url=None, api="chat/completions")` | OpenAI-compatible API | also works with local OpenAI-compatible servers through `base_url`; costs money per call |

`max_new_tokens` defaults to 32, which cuts long answers short. Raise it for anything but short
factoid answers.

For FiD readers, agentic search and other advanced components, read
[references/advanced.md](references/advanced.md).

## Datasets and corpora

```python
nq = pt.get_dataset("rag:nq")
topics = nq.get_topics("dev")          # qid, query
answers = nq.get_answers("dev")        # qid, gold_answer: one row per acceptable answer
```

- `rag:nq`, `rag:hotpotqa`, `rag:triviaqa`, `rag:web_questions`, `rag:wow`, `rag:popqa`,
  `rag:2wikimultihopqa` come from the FlashRAG datasets on Hugging Face. Splits are `train`, `dev` and
  sometimes `test`.
- **Always pass the split.** `get_answers()` without it fails with
  `AttributeError: 'FlashRAGDataset' object has no attribute 'name'`.
- Corpora: `rag:nq_wiki` (Wikipedia 2018 passages) and `rag:hotpotqa_wiki`. Prebuilt indexes:
  `pyterrier/ragwiki-terrier` (13 GB, BM25, stores text), `pyterrier/ragwiki-e5.flex` (65 GB, E5),
  `pyterrier/raghpq-terrier` (see the `pyterrier-huggingface` skill). Use the corpus the dataset was
  built for.
- For a CPU demo on `rag:nq`, the BEIR index `pt.Artifact.from_hf("pyterrier/nq.terrier")` (1.4 GB, BM25,
  stores text) covers the same NQ Wikipedia passages. Its docnos differ from `rag:nq_wiki`: harmless for
  `EM`/`F1`, which score per `qid`, but wrong for anything that needs the dataset's own docno space.

## Evaluating answers

Pass the answers in place of qrels:

```python
results = pt.Experiment(
    [rag_bm25, rag_e5],
    topics, answers,
    eval_metrics=[ptr.measures.EM, ptr.measures.F1],
    names=["BM25 >> reader", "E5 >> reader"],
)
```

| Measure | What it scores |
|---|---|
| `ptr.measures.EM` | normalised exact match against any gold answer |
| `ptr.measures.F1` | token-overlap F1, best over gold answers |
| `ptr.measures.ROUGE1F`, `ROUGE2F`, `ROUGELF` (and `P`/`R` variants) | n-gram overlap, for long answers |
| `ptr.measures.BERTScore(...)` | semantic similarity; downloads a model |
| `ptr.measures.AnswerLen`, `AnswerZeroLen` | answer length, and the share of empty answers |
| `ptr.measures.LLMasJudge` | an LLM grades the answer |

- Evaluate retrieval separately too. Answer quality is capped by whether the passages contain the
  answer.
- Report the reader model, prompt, `k`, context limits and `max_new_tokens`.

## Don't

- Don't use the default `Reader` prompt with a model FastChat doesn't know. Use `raw_instruction=True`
  and inspect the `prompt` column.
- Don't call `get_answers()` without a split.
- Don't keep `max_new_tokens=32` for long-form answers.
- Don't compare readers with different `k` or context limits without saying so.
- Don't download the 65 GB `ragwiki-e5.flex` without checking disk space and asking.
- Don't install pyterrier-generative into this environment. It replaces pyterrier-rag with a branch
  build (see the `pyterrier-generative` skill).

## Reference files

| Read | When |
|---|---|
| [references/advanced.md](references/advanced.md) | FiD readers, agentic search (SearchR1, SearchO1, R1Searcher), LLM-as-judge, custom prompts |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Answers are `'### Assistant'` or unrelated text | FastChat fallback template for a non-chat model | `PromptTransformer(instruction=template, raw_instruction=True)` |
| `TypeError: expected string or bytes-like object, got 'DataFrame'` from `rag.search(...)` | `search()` takes one query string | `rag.transform(topics)` |
| `AttributeError: 'FlashRAGDataset' object has no attribute 'name'` | `get_answers()` without a split | `get_answers("dev")` |
| Answers cut off mid-sentence | `max_new_tokens=32` default | raise `max_new_tokens` on the backend |
| EM and F1 are 0 | the corpus doesn't contain the answers, or answers aren't normalised | check retrieval recall; use the dataset's own corpus |
| Context loses the best passages | prompt truncated to `max_input_length` | lower `k` or set `Concatenator(max_length=..., tokenizer=...)` |
| `ModuleNotFoundError: vllm` | `VLLMBackend` without vLLM | `pip install vllm` (GPU), or use `HuggingFaceBackend` |
| `RuntimeError: operator torchvision::nms does not exist` on import | installing pyterrier-rag pulled a CUDA `torchvision` (and `torchaudio`) next to a CPU-only `torch`, or a `torchvision` from another environment leaks in through `.pth` or `--system-site-packages` (see the `pyterrier-colbert` skill) | `pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cpu`, and remove or match `torchaudio`; or build a fully isolated environment |
