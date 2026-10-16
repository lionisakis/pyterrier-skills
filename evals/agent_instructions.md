# Instructions for a skill evaluation run

You are a coding agent with the pyterrier-* skills installed in this project's `.claude/skills/`.
A user has typed the task below. Do the task for them, the way you normally would, and then
write a findings report about the skills themselves.

Python: `{python}` (pyterrier 1.x, Java 11, CPU only unless the task says otherwise).
Workspace (write everything here; indexes land here): `{workspace}`

## Rules

- Role-play honestly: open the skills whose descriptions match the task, read SKILL.md first, follow
  its pointers to `references/` only when it tells you to, and write the code the skill leads you to.
- If the skill leads you into an error, record the error text, THEN fix it from the installed source
  (`inspect`, `help()`, reading site-packages) and record what you needed that the skill did not say.
- Do not consult the PyTerrier website first. The point is to measure whether the skill alone is
  enough. After a failure you may consult it, and must say so.
- Do not modify any file under `.claude/skills/`; propose edits in the report instead.
- CPU only: set `OMP_NUM_THREADS=8` and `TOKENIZERS_PARALLELISM=false`; keep corpora and models small;
  wrap each run in `timeout 900`.
- Save every script you write in the workspace so a human can re-run it.
- When a task compares systems in `pt.Experiment`, pass `baseline=` and `correction="bonferroni"` unless the
  task names another correction, so runs are comparable across evals.

## The user's task

{prompt}

## Report

Write `{workspace}/findings.md` with exactly these sections:

1. **Task outcome** — did the task succeed? Final numbers or results table.
2. **Skill verdict per skill used** — 1-3 sentences each: sufficient or not, best and worst part.
3. **Defects** — table: `# | skill | file:line | what the skill says | what is true (evidence) | severity (blocker/wrong/unclear/missing)`.
4. **Friction** — correct but costly: ambiguous pointers, a missing example, text in the wrong file.
5. **Outside knowledge needed** — what you had to get from source or memory that the skill should say.
6. **Proposed edits** — minimal edits in the repo's style (a "Don't" bullet, a Troubleshooting row, or a
   replacement sentence or snippet). Quote the exact old text and the exact new text.
7. **Trigger check** — from the descriptions alone, which skills would you open for this task? Any
   opened and not needed, or needed and not opened?
8. **Run log** — commands, wall time, and the last ~30 lines of output of each.

Your final message should contain sections 1, 3 and 6 only.
