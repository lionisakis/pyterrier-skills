# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Agent skills that teach coding agents to develop with [PyTerrier](https://github.com/terrier-org/pyterrier)
and its extensions. Skills follow the open [Agent Skills](https://agentskills.io/specification) format and
are installed locally with the cross-agent `npx skills` CLI. There is no plugin marketplace and nothing is
published. `pyproject.toml` only defines the dev environment and the test suite that checks skills.

`skills/` is currently empty: the first skills have not been written yet.

## Layout

```
skills/<skill-name>/SKILL.md   # one folder per skill, plus optional references/ and scripts/
tests/skill_files.py           # helpers: find skills, parse frontmatter, extract code blocks
tests/test_skill_format.py     # frontmatter, size and link rules
tests/test_snippets.py         # python blocks must parse; `python run` blocks execute
pyproject.toml                 # dev dependency group + pytest config (not a package)
```

Never put a `SKILL.md` anywhere except `skills/<name>/SKILL.md`. The installer scans the repo root and
nested folders, and a shallower `SKILL.md` shadows deeper ones; `test_no_stray_skill_files` enforces this.

## Commands

```bash
uv sync                          # or: python3 -m venv .venv && .venv/bin/pip install --group dev  (pip >= 25.1)
uv run pytest                    # fast: format checks + every python block parses
uv run pytest -m run             # execute `python run` blocks (needs pyterrier[all] and Java 11+)
uv run pytest -m slow            # execute run blocks including `python run slow` ones
uv run pytest -m run -k <skill>  # one skill
npx skills add . --list          # preview what the installer finds
npx skills add . --agent claude-code   # install into ./.claude/skills of the current project
```

Run `npx skills add` from the project that should receive the skills, passing this repo's path. From a
local path it copies each skill into `.claude/skills/` and records it in `skills-lock.json`, so re-run the
command after editing a skill. `-g` installs for the user instead of the project. The CLI needs Node 18+.

## Adding a skill

1. `npx skills init skills/<name>`, or create `skills/<name>/SKILL.md` by hand.
2. Frontmatter: `name` (must equal the folder name; lowercase letters, digits, single hyphens; max 64),
   `description` (max 1024, written as a trigger condition: *when* to use the skill), and
   `compatibility` (the versions it was checked against, e.g. `pyterrier>=1.1`).
3. Fence examples that must actually work as ```` ```python run ```` (or `python run slow` for large
   downloads/GPU). Run blocks in one file share a namespace and execute in order. Plain ```` ```python ````
   blocks may be fragments but must still parse.
4. Add any new dependency (e.g. `pyterrier-dr`) to the `dev` group in `pyproject.toml`.
5. `uv run pytest`, then `uv run pytest -m run -k <name>`.

## Writing rules

- Keep `SKILL.md` under 500 lines. Move API tables and long examples into `references/`, and say in the
  pointer *when* to read each file.
- Open with links to the relevant pages on https://pyterrier.readthedocs.io/en/latest/ and close with a
  Troubleshooting table (symptom, cause, fix).
- Prefer tables and code to prose. State mistakes as a flat "Don't" list, not graded by severity.
- When an agent gets something wrong in testing, add one concrete correction to the Don't list or
  Troubleshooting table rather than another paragraph.
- Refer to other skills by name, never by a relative path into another skill's folder.

## PyTerrier content rules

- **Target PyTerrier 1.x.** Never emit legacy APIs: `pt.init()`, `pt.BatchRetrieve` (use
  `pt.terrier.Retriever` or `TerrierIndex(...).bm25()`), `pyt_splade` (use `pyterrier_splade`), the `~`
  caching operator (use pyterrier-caching), `pyterrier.transformer.TransformerBase`.
- **Do not copy doc snippets unchecked.** The official docs contain broken examples (wrong operator
  precedence on the operators page, `"c"` instead of `"bm25.b"` in tuning, `docid` instead of `docno` in the
  pyterrier-dr how-to, and others). Prove examples with `python run` blocks.
- **Operator precedence follows Python:** `**` > `*` `%` > `+` > `>>` > `&` > `^` > `|`. Parenthesise `&`, `|`, `^`.
- **Code is illustrative, not prescriptive.** Collection, topics, rerank depth and metrics come from the
  user's task or the paper being reproduced, never from an example. Small collections such as `vaswani` are
  fine for checking that code runs, never for comparing against published numbers.
- **Prefer prebuilt indices** via `pt.Artifact.from_hf(...)` over building new ones.
- **Metric correctness over brevity.** For TREC DL, nDCG@10 uses graded labels and MAP/MRR binarise at
  rel >= 2; MS MARCO dev uses `RR@10` on binary labels. BM25 defaults differ: Terrier k1 1.2, b 0.75;
  Anserini MS MARCO 0.82, 0.68; PISA 0.9, 0.4.
- **Reproductions report faithfully:** separate published and reproduced columns, never adjust evaluation
  to move results, record assumptions.

## Removed first-draft skills

Three earlier skills were removed so new ones can be written to these rules. Read them for inspiration with
`git show 1ce58e5:<file>`: `msmarco-passage-skill.md`, `reproduction-skill.md`,
`reranking-experiments-skill.md`. Do not copy their bugs:

- The DL'20 entry uses the `trec-dl-2019` dataset id; it should be `irds:msmarco-passage/trec-dl-2020/judged`.
- The reranker class reads `group["scores"]` where it means the `docno` column.
- It uses `pta.DataFrameBuilder` without importing pyterrier-alpha; use `pt.new.DataFrameBuilder` (core since 1.1).
- The `pt.Experiment` examples are missing a comma after the list of systems.
