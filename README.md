# pyterrier-skills

Agent skills for developing with [PyTerrier](https://github.com/terrier-org/pyterrier) and its extensions,
in the open [Agent Skills](https://agentskills.io/specification) format. They work with Claude Code,
Cursor, Codex, Gemini CLI and other agents supported by the [`skills` CLI](https://github.com/vercel-labs/skills).

> Status: in progress. This repository is private and not published.

## Skills

Skills are organised by pipeline stage. `pyterrier-basics` is the base skill: install it with any of the others.

| Skill | Covers | Status |
|---|---|---|
| `pyterrier-basics` | how PyTerrier works, install, data model, operators, `pt.apply`, custom transformers, legacy → 1.x, building a pipeline | done |
| `pyterrier-experiments` | datasets, `pt.Experiment`, significance, benchmark conventions, SuiteEval, tuning | planned |
| `pyterrier-huggingface` | finding, loading and sharing prebuilt indexes and caches | planned |
| `pyterrier-first-stage` | Terrier, PISA, SPLADE, dense retrieval, ColBERT v2, fusion | planned |
| `pyterrier-expansion` | query rewriting, PRF, document expansion (doc2query) | planned |
| `pyterrier-reranking` | cross-encoders, T5, monoELECTRA, LLM rerankers, learning to rank | planned |
| `pyterrier-rag` | retrieval-augmented generation | planned |
| `pyterrier-caching` | caching expensive pipeline stages | planned |

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
