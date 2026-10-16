# Skill evaluations

`pytest` proves that the code in the skills runs. It cannot tell whether an agent that reads a skill
writes the right code for a real task. These evals do that: each entry in `evals.json` is a realistic
user prompt that exercises one or more skills, run through the `claude` CLI with every skill installed.
The agent does the task, then writes `findings.md` listing defects, friction and proposed edits.

## Loop

1. `python evals/run_evals.py --list` to see the prompts. Add one when a skill gains a feature or a
   user hits a problem that no prompt covers.
2. `python evals/run_evals.py --model opus --jobs 3` (one iteration; results land in
   `../pyterrier-skills-evals/iteration-N/eval-<id>-<name>/`).
3. Read `iteration-N/report.md` (written at the end of the batch; `python evals/report.py` rebuilds it):
   wall, LLM and tool time, human messages and LLM turns, tokens, CPU time and peak memory, GPU,
   prebuilt/cache reuse vs index builds, skill vs library-source reads (`evals/source_reads.py
   --details` lists them), pt.Experiment baseline and correction, and overlap of the outcome numbers
   with the previous iteration. Many source reads on one topic mean the skill is missing something.
   Then read each `findings.md`. For every defect, apply the smallest correction in CLAUDE.md's style: a
   "Don't" bullet, a Troubleshooting row, or a fixed snippet. Do not add paragraphs.
4. `uv run pytest`, then `uv run pytest -m run -k <skill>` for any skill whose `python run` blocks changed.
5. Re-run the affected eval (`--id N`) and check the defect is gone from the new findings.

`agent_instructions.md` is the prompt template; `{prompt}`, `{python}` and `{workspace}` are filled in by
the runner. The `expectations` in `evals.json` are the checks a grader applies to the run; they are
read by a human or a grading agent, not by pytest.

`--max-turns` (default 150) applies to every eval. An eval that needs more sets its own `"max_turns"` in
`evals.json` (eval 8 has 300); the runner records it in `eval_metadata.json` and the report uses it.

The workspace is created outside the repo on purpose: a copied `SKILL.md` inside the repo would be
reported by `test_no_stray_skill_files`.

## Pipeline matrix (resource ablation)

A second family asks how much skills, documentation and code view each help an agent build a whole
experiment from scratch. One agent per condition creates its own venv, builds every index (no prebuilt
ones) and runs 9 pipelines (BM25, BM25>>RM3>>BM25, SPLADE, RetroMAE, SPLADE>>monoT5, RetroMAE>>monoT5,
BM25>>RM3>>BM25>>SPLADE>>monoT5, BM25>>ColBERT>>ColBERT-PRF>>ColBERT, BM25 score injected into a
cross-encoder) on MS MARCO dev, DL19, DL20 and the 13 NanoBEIR subsets. The prompt does not mention
SuiteEval. Everything is defined in `pipelines/pipelines.json`; the prompt is `pipelines/agent_instructions.md`.

| condition | skills | docs | code view |
|---|---|---|---|
| none, code, docs, skills, docs_code, skills_code, skills_docs, all | `.claude/skills/` | `pyterrier-docs/` + `search_docs.py` + web | read the venv's site-packages, inspect, GitHub |

`pipelines/block_source.py` (a PreToolUse hook) enforces each condition and logs blocks to
`hook_blocks.jsonl`. Every condition is blocked from this repo and from the other conditions' folders.
Runs need GPU nodes, Java 11+ and a lot of disk for the MS MARCO indexes: point `--workspace` at
node-local storage.

```bash
python evals/fetch_docs.py                                   # once: mirror the docs into pyterrier-docs/ (gitignored)
python evals/run_pipelines.py --model qwen-3.8-27b --effort medium --dry-run
python evals/run_pipelines.py --model qwen-3.8-27b --effort medium            # all 8 conditions, one at a time
python evals/run_pipelines.py --model qwen-3.8-27b --effort medium --conditions none,code,docs,skills --iteration pipelines-1
python evals/run_pipelines.py --model qwen-3.8-27b --effort medium --conditions all --pipelines bm25,bm25_prf_bm25 --datasets dl19   # smoke test
bash evals/pipelines/run_without_skills.sh                 # none, code, docs, docs_code into the smoke batch; one background process each
python evals/pipeline_status.py --watch 60 --deadline 23:30  # pipelines × datasets (x/8 conditions done, ▶ active, ✗ failed) + ETA
python evals/pipeline_status.py --condition skills           # one condition's cells
python evals/grade_pipelines.py --controls                   # pt.Experiment per condition × dataset, BM25 baseline, Bonferroni; agreement vs `all`
python evals/pipeline_report.py                              # one row per condition: time, interaction, tokens, CPU/GPU, reuse, resources, quality
```

When an agent stops with cells missing, the runner resumes it with a fixed nudge (up to `--max-nudges`,
default 3); each nudge counts as a human message. A cell is done only when its run file
`runs/<slug>/<pipeline>.res.gz` parses and covers at least 90% of the topics, whatever `status.json` says.
The gateway puts idle models to sleep. Before each attempt the runner wakes the model if it is not running (up to
`--wake-timeout-min`, default 45). While a condition runs, it pings the model every `--keepalive-min` minutes (default 10;
0 = off). API errors are retried even when all cells are done, so the agent can still evaluate and write its report.
Pods are culled after about 10 minutes of low CPU. To prevent that, the runner keeps one core busy for `--cpu-heartbeat-s` seconds each
minute (default 10, about 0.17 cores; 0 = off). That CPU is not counted as the agent's. Use `2>&1 | tee <log>` instead of `> <log>`, so
the 60 s heartbeat lines also show up in the terminal.
If a condition was killed, use `--resume --iteration pipelines-N --conditions <id>` to continue its last session. The agent is
told which cells are missing. Its files are kept, and the time it ran before being killed is added to `timing.json`.
