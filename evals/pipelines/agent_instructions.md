# Build and evaluate a set of PyTerrier pipelines

You are a coding agent. A user has asked you to build the experiment below with PyTerrier, end to end:
create the Python environment, build every index yourself, run every pipeline on every dataset and
write the run files. Work without asking questions; nobody will answer until you stop.

Workspace (write everything here): `{workspace}`
Base Python for creating your environment: `{base_python}`
Hardware: {hardware}

## What you can use

{resources}

## The experiment

Pipelines (use these ids in file names; use the listed models, and record any substitution):

{pipelines}

Datasets (ir_datasets ids; PyTerrier can load them as `pt.get_dataset("irds:<id>")`):

{datasets}

{dataset_notes}Work only on the datasets listed above.

You may download prebuilt indexes, but only from Hugging Face (huggingface.co); build any index you
do not take from there. Model checkpoints may be downloaded.

## Output contract (graded by a script, so follow it exactly)

1. Create your environment in `{workspace}/venv` and keep indexes under `{workspace}/indexes/`.
2. Write every run (one file per pipeline per dataset) to
   `{workspace}/runs/<slug>/<pipeline>.res.gz` in TREC run format (`pt.io.write_results`), where
   `<slug>` is the ir_datasets id with `/` replaced by `-`, e.g.
   `runs/{example_slug}/bm25.res.gz`.
   Keep the dataset's original qids and docnos. Return up to 1000 documents per query (100 for
   re-rankers).
3. Keep `{workspace}/status.json` up to date whenever a cell starts, finishes or fails:
   `{{"<pipeline>": {{"<dataset id>": {{"state": "building|running|done|failed", "updated": "<ISO time>", "note": "..."}}}}}}`
   with dataset ids {status_ids}.
4. Write `{workspace}/manifest.json`:
   `{{"env_setup_s": ..., "indexes": [{{"name": ..., "type": "terrier|splade|dense|colbert|...", "corpus": ..., "path": ..., "build_s": ...}}],
   "pipelines": {{"<pipeline>": {{"models": [...], "indexes": [...], "notes": "..."}}}}}}`.
5. Save every script you run in `{workspace}/scripts/` and every log in `{workspace}/logs/`.
6. Evaluate what you built: for each dataset, one `pt.Experiment` over your runs with BM25 as
   `baseline=`, `correction="bonferroni"`, and the measures {measures}. Save the tables to
   `{workspace}/results/`.
7. Finish with `{workspace}/final_report.md`: the result tables, which cells failed and why, and
   anything you substituted.

## How to work

- Commands that take more than a few minutes (installs, indexing, encoding, re-ranking): start them
  with `nohup ... > {workspace}/logs/<name>.log 2>&1 &`, then poll the log. A single tool call cannot
  run for more than 10 minutes.
- Reuse what you have built: one index per corpus per index type, and caches for re-rankers and
  first stages that several pipelines share. Building the same thing twice counts against you.
- Do the cheap cells first (BM25 on DL19/DL20, NanoBEIR), then the MS MARCO index and MS MARCO dev.
- If a cell cannot be done, mark it `failed` in `status.json` with a note and move on.

## The user's request

{prompt}
