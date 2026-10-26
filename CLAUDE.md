# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Agent skills that teach coding agents to develop with [PyTerrier](https://github.com/terrier-org/pyterrier)
and its extensions. Skills follow the open [Agent Skills](https://agentskills.io/specification) format and
are installed with the cross-agent `npx skills` CLI, straight from the GitHub repo; there is no registry.
`pyproject.toml` defines the dev environment and the test suite, and builds a docs-only wheel that carries
`pt_docs/` for the PyTerrier docs (no Python code, no skills).

Skills come in two layers (see README.md): stage skills (retrieval, fusion, expansion, reranking, rag) that
compare options and route to the model-family skills (splade, dense, colbert, generative), which cover every role of one
model type. `pyterrier-basics` is the base skill:
every other skill opens with "Assumes the `pyterrier-basics` skill." and does not repeat its content
(install, data model, operators, legacy → 1.x).

## Layout

```
skills/<skill-name>/SKILL.md   # one folder per skill, plus optional references/ and scripts/
tests/skill_files.py           # helpers: find skills, parse frontmatter, extract code blocks
tests/test_skill_format.py     # frontmatter, size and link rules
tests/test_snippets.py         # python blocks must parse; `python run` blocks execute
pyproject.toml                 # dev dependency group, pytest config, docs-only wheel (pt_docs/)
evals/evals.json               # realistic task prompts; evals/run_evals.py runs them through `claude -p`
evals/pipelines/               # pipeline-matrix evals (skills/docs/code-view ablation); evals/run_pipelines.py
ext-skills/<skill-name>/       # GENERATED: skills copied from extensions' pt_skills/; never edit by hand
ext-skills/manifest.json       # where each ext-skill came from, and whether it replaces a skills/ skill
scripts/collect_ext_skills.py  # fills ext-skills/ from PyTerrier's docs/extensions.txt (weekly Action opens a PR)
pt_docs/                       # this repo's page in the PyTerrier docs; skills.rst is generated
scripts/build_pt_docs.py       # regenerates pt_docs/skills.rst from the frontmatter
.github/workflows/             # collect-ext-skills.yml (weekly) and install-check.yml (after skills change)
```

Never put a `SKILL.md` anywhere except `skills/<name>/SKILL.md` (and the generated
`ext-skills/<name>/SKILL.md`). The installer scans the repo root and nested folders, and a shallower
`SKILL.md` shadows deeper ones; `test_no_stray_skill_files` enforces this.

## Extension skills

Extensions can ship skills in `<package>/pt_skills/<name>/SKILL.md`, the way they ship `pt_docs/`
(guide for authors: `pt_docs/authoring.rst`). `python scripts/collect_ext_skills.py` reads PyTerrier's
`docs/extensions.txt` and copies every such folder into `ext-skills/` (`--from github`, the default, reads
each repo's default branch; `--from installed` reads pip-installed packages like the docs build does). A
collected skill replaces the `skills/` skill of the same name everywhere: `find_skills()`, the evals and
the install. Fix a collected skill upstream in its extension, not here. After any skill or frontmatter
change, run `python scripts/build_pt_docs.py` (a test fails if `pt_docs/skills.rst` is stale).

## Commands

```bash
uv sync                          # or: python3 -m venv .venv && .venv/bin/pip install --group dev  (pip >= 25.1)
uv run pytest                    # fast: format checks + every python block parses
uv run pytest -m run             # execute `python run` blocks (needs pyterrier[all] and Java 11+)
uv run pytest -m slow            # execute run blocks including `python run slow` ones
uv run pytest -m run -k <skill>  # one skill
python evals/run_evals.py --model opus   # agents do real tasks with the skills and report defects (see evals/README.md)
npx skills add . --list          # preview what the installer finds
npx skills add . --agent claude-code   # install into ./.claude/skills of the current project
python scripts/collect_ext_skills.py     # refresh ext-skills/ (the weekly Action does this and opens a PR)
python scripts/build_pt_docs.py          # regenerate pt_docs/skills.rst
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
5. `python scripts/build_pt_docs.py`, `uv run pytest`, then `uv run pytest -m run -k <name>`.

## Improving a skill from agent runs

`pytest` proves the snippets run; it cannot tell whether an agent that reads a skill writes the right
code for a real task. `evals/evals.json` holds realistic user prompts, one per skill group.
`python evals/run_evals.py --model opus` runs each through the `claude` CLI with every skill installed
and collects a `findings.md` per prompt: defects with file:line evidence, proposed edits, and a trigger
check against the descriptions. Turn each defect into one "Don't" bullet or Troubleshooting row, run
`pytest`, then re-run that eval with `--id N`. Add a prompt whenever a user hits a problem that no
existing prompt covers. Details in `evals/README.md`.

## Writing rules

- Keep `SKILL.md` under 500 lines. Move API tables and long examples into `references/`, and say in the
  pointer *when* to read each file.
- Open with links to the relevant pages on https://pyterrier.readthedocs.io/en/latest/ and close with a
  Troubleshooting table (symptom, cause, fix).
- Prefer tables and code to prose. State mistakes as a flat "Don't" list, not graded by severity.
- When an agent gets something wrong in testing, add one concrete correction to the Don't list or
  Troubleshooting table rather than another paragraph.
- Refer to other skills by name, never by a relative path into another skill's folder.
- Write the `description` in the third person: what the skill covers and when to use it, with trigger
  keywords and a "Not for …" clause where another skill is a near miss. Don't summarise the steps.
- Code in `references/` is not executed by the tests. Run it yourself before committing.

## PyTerrier content rules

- **Target PyTerrier 1.x.** Never emit legacy APIs: `pt.init()`, `pt.BatchRetrieve` (use
  `pt.terrier.Retriever` or `TerrierIndex(...).bm25()`), `pyt_splade` (use `pyterrier_splade`), the `~`
  caching operator (use pyterrier-caching), `pyterrier.transformer.TransformerBase`.
- **Do not copy doc snippets unchecked.** The official docs contain broken examples (wrong operator
  precedence on the operators page, missing commas in the tuning example, `docid` instead of `docno` in the
  pyterrier-dr how-to, `pt.validate.query_frame(input_columns, ...)` inside `transform(inp)` on the inspection
  troubleshooting page, and others). Prove examples with `python run` blocks. (The tuning page's `"c"` control
  is not a bug: in Terrier 5.11 it sets the same BM25 `b` as `"bm25.b"`. Skills still use `"bm25.b"`.)
- **Operator precedence follows Python:** `**` > `*` `%` > `+` > `>>` > `&` > `^` > `|`. Parenthesise `&`, `|`, `^`.
- **Code is illustrative, not prescriptive.** Collection, topics, rerank depth and metrics come from the
  user's task or the paper being reproduced, never from an example. Small collections such as `vaswani` are
  fine for checking that code runs, never for comparing against published numbers.
- **Prefer prebuilt indices** via `pt.Artifact.from_hf(...)` over building new ones.
- **Metric correctness over brevity.** For TREC DL, nDCG@10 uses graded labels and MAP/MRR binarise at
  rel >= 2; MS MARCO dev uses `RR@10` on binary labels. BM25 defaults differ: Terrier k1 1.2, b 0.75;
  Anserini MS MARCO 0.82, 0.68; PISA 0.9, 0.4.
- **Report faithfully:** when comparing with published numbers, show them in a separate column with the
  difference, never adjust evaluation to move results, and record assumptions. Skills teach how to create
  pipelines and experiments; none is framed as reproducing a paper.
- **Parameter defaults live in `references/` only**, with an instruction to verify them against the installed
  source, the index metadata or the paper.

## Removed first-draft skills

Three earlier skills were removed so new ones can be written to these rules. Read them for inspiration with
`git show 1ce58e5:<file>`: `msmarco-passage-skill.md`, `reproduction-skill.md`,
`reranking-experiments-skill.md`. Do not copy their bugs:

- The DL'20 entry uses the `trec-dl-2019` dataset id; it should be `irds:msmarco-passage/trec-dl-2020/judged`.
- The reranker class reads `group["scores"]` where it means the `docno` column.
- It uses `pta.DataFrameBuilder` without importing pyterrier-alpha; use `pt.new.DataFrameBuilder` (core since 1.1).
- The `pt.Experiment` examples are missing a comma after the list of systems.
