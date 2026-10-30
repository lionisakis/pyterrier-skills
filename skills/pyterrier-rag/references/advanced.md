# Advanced RAG components

Read from the source of pyterrier-rag 0.3.0. These components need large models or a GPU (7B models,
vLLM), so only the class signatures were checked here, not runs. Test on two or three queries with
`.search()` before a full run.

## Contents
1. Fusion-in-Decoder readers
2. Agentic search
3. Custom prompts
4. LLM-as-judge

## 1. Fusion-in-Decoder readers

FiD encodes each passage separately with the question and fuses them in the decoder. That is
different from `Concatenator` + `Reader`: FiD takes the results directly, with **no** `Concatenator`.

```python
from pyterrier_rag.readers import T5FiD, BARTFiD

fid = T5FiD("<a FiD checkpoint>", batch_size=4, num_context="auto", text_max_length=256, max_new_tokens=32)
rag = retriever % 10 >> dataset.text_loader() >> fid
```

`T5FiD(model_name_or_path, tokenizer_name_or_path=None, batch_size=4, text_field="text",
text_max_length=256, num_context="auto", max_new_tokens=32, generation_config=None, verbose=False,
device=None)`. `BARTFiD` is the BART version. Use a checkpoint trained for FiD. A plain T5 checkpoint
won't produce good answers.

## 2. Agentic search

These models decide when to search and what to search for, interleaving reasoning and retrieval.
Each takes a PyTerrier retriever that returns `text` (and usually `title`) for a query.

| Class | Default model | Signature (main arguments) |
|---|---|---|
| `ptr.SearchR1` | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo` | `SearchR1(retriever, model_id=..., retrieval_top_k=3)` |
| `ptr.SearchO1` | (a `HuggingFaceBackend` you pass) | `SearchO1(retriever, generator, max_turn=10, max_retrieval_step=5, topk=10, temperature=0.7, ...)` |
| `ptr.R1Searcher` | `XXsongLALA/Qwen-2.5-7B-base-RAG-RL` | `R1Searcher(retriever, model_path=..., model_kw_args={...}, top_k=5)`; needs **vLLM** |

```python
agent = ptr.SearchR1(index.bm25() >> dataset.text_loader(), retrieval_top_k=3)
answers = agent(topics)          # qanswer, plus the reasoning trace
```

- The default models are 7B and need a GPU with enough memory. R1Searcher requires `vllm`.
- The retriever runs many times per question. Cache it (see the `pyterrier-caching` skill).

## 3. Custom prompts

```python
from pyterrier_rag.prompt import PromptTransformer
from pyterrier_rag.readers import Reader

prompt = PromptTransformer(
    instruction="Answer in one short phrase.\nContext: {{ qcontext }}\nQuestion: {{ query }}\nAnswer:",
    raw_instruction=True,              # send exactly this text (base or seq2seq models)
)
reader = Reader(backend, prompt=prompt)
```

`PromptTransformer(instruction, model_name_or_path=None, system_message=None, conversation_template=None,
api_type=None, output_field="prompt", input_fields=["query", "qcontext"], expects_logprobs=False,
answer_extraction=None, raw_instruction=False)`.

- Without `raw_instruction=True`, the instruction is wrapped in a FastChat conversation template.
  It's chosen from `model_name_or_path` (or from the backend's model when `Reader` builds the prompt),
  with a one-shot fallback for unknown models. See SKILL.md, "Prompts".
- `system_message=` sets the system turn for chat models.
- `answer_extraction=` takes a function that cleans the model output (e.g. removes "Answer:").
- `input_fields` lists the columns the template uses. Add any extra column you reference.

## 4. LLM-as-judge

`ptr.measures.LLMasJudge` grades answers with an LLM. Read its docstring
(`help(ptr.measures.LLMasJudge)`) for its arguments. Judges cost as much as generation, can be biased
towards their own model family, and should be reported with the judge model and prompt. Report EM and
F1 alongside them.
