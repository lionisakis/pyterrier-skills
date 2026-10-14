# pyterrier-skills

Agent skills for developing with [PyTerrier](https://github.com/terrier-org/pyterrier) and its extensions,
in the open [Agent Skills](https://agentskills.io/specification) format. They work with Claude Code,
Cursor, Codex, Gemini CLI and other agents supported by the [`skills` CLI](https://github.com/vercel-labs/skills).

> Status: all twelve skills written and tested (CPU only). This repository is private and not published.

## Skills

Skills come in two layers. Stage skills (`pyterrier-retrieval`, `pyterrier-fusion`, `pyterrier-expansion`,
`pyterrier-reranking`, `pyterrier-rag`) say what each stage does and compare the options; model-family skills
(`pyterrier-splade`, `pyterrier-dense`, `pyterrier-colbert`, `pyterrier-generative`) cover every role of one model type, as retriever,
reranker or expansion. `pyterrier-basics` is the base skill: install it with any of the others.

| Skill | Covers | Status |
|---|---|---|
| `pyterrier-basics` | how PyTerrier works, install, data model, operators, `pt.apply`, custom transformers, legacy → 1.x, building a pipeline | done |
| `pyterrier-experiments` | datasets, `pt.Experiment`, significance, benchmark conventions, SuiteEval, tuning | done |
| `pyterrier-huggingface` | finding, loading and sharing prebuilt indexes and caches | done |
| `pyterrier-retrieval` | Terrier, PISA, BM25 parameters, query punctuation, choosing a first-stage family | done |
| `pyterrier-fusion` | reciprocal rank fusion, score interpolation, fusing retrievers, rerankers and saved runs | done |
| `pyterrier-splade` | SPLADE: encoding, PISA/Terrier indexes, retrieval, scorer, prebuilt vectors | done |
| `pyterrier-dense` | pyterrier-dr: encoders, FlexIndex, exact/ANN retrieval, scorer, vector PRF | done |
| `pyterrier-colbert` | late interaction: ColBERT v2 / PLAID (index, retrieval, text scorer, PLAID-PRF) and PyLate (pyterrier-pylate) | done (pilot of the model-family layout) |
| `pyterrier-expansion` | query rewriting, PRF, document expansion (doc2query) | done |
| `pyterrier-reranking` | cross-encoders, T5, monoELECTRA, learning to rank; routes LLM rerankers to `pyterrier-generative` | done |
| `pyterrier-generative` | LLM listwise/setwise rerankers (RankZephyr, RankVicuna, RankGPT, custom prompts) from pyterrier-generative 0.1.5 on GitHub | done (checked in a separate venv; not in the dev group, it conflicts with pyterrier-rag) |
| `pyterrier-rag` | retrieval-augmented generation | done |
| `pyterrier-caching` | caching expensive pipeline stages | done |

## Install locally

From the project where you want the skills, point the installer at a checkout of this repo:

```bash
npx skills add /path/to/pyterrier-skills --list                 # preview
npx skills add /path/to/pyterrier-skills --agent claude-code    # install for Claude Code
```

The skills are copied into the project's `.claude/skills/` and recorded in `skills-lock.json`. Re-run the
command to pick up changes from this checkout. Use `-g` to install them for your user instead of the
project. The `skills` CLI needs Node 18 or newer.

## Develop

```bash
uv sync               # or: python3 -m venv .venv && .venv/bin/pip install --group dev   (pip >= 25.1)
uv run pytest         # format checks, and every python block parses
uv run pytest -m run  # execute `python run` blocks (needs Java 11+ for Terrier)
uv run pytest -m slow # also execute `python run slow` blocks
```

See [CLAUDE.md](CLAUDE.md) for how to write a skill.
