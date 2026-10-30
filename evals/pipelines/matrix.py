"""Shared helpers for the pipeline-matrix evals: the spec, the cells, run-file checks and cell states.

Layout of one batch (`<workspace>/pipelines-N/`):
    run_meta.json                 model, effort, flags; one entry per node that ran part of it
    <condition>/                  the agent's workspace (venv/, indexes/, runs/, status.json, ...)
        project/                  the agent's cwd (.claude/skills, pyterrier-docs, search_docs.py)
        cells.json                the cells this condition was asked for
        agent_state.json          queued | running | finished, attempts, exit codes (written by the runner)
        status_log.jsonl          every cell state change the runner saw in status.json, with time
        timing.json, result.json  as in run_evals.py, summed over the first run and every nudge
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "pipelines.json"
REPO = HERE.parent.parent
DEFAULT_WORKSPACE = REPO.parent / f"{REPO.name}-evals"
COLUMNS = ["msmarco-dev", "dl19", "dl20", "nanobeir"]
STATES = ("pending", "building", "running", "done", "failed")


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def slug(irds: str) -> str:
    return irds.replace("/", "-").lower()


def all_cells(spec: dict, pipelines: list[str] | None = None, datasets: list[str] | None = None) -> list[dict]:
    """One dict per (pipeline, ir_datasets id). NanoBEIR expands to its 13 subsets, column 'nanobeir'.
    A pipeline with a `datasets` list in the spec runs only on those columns."""
    out = []
    for p in spec["pipelines"]:
        if pipelines and p["id"] not in pipelines:
            continue
        for d in spec["datasets"]:
            if datasets and d["id"] not in datasets:
                continue
            if p.get("datasets") and d["id"] not in p["datasets"]:
                continue
            for irds in d.get("subsets") or [d["irds"]]:
                key = irds if d.get("subsets") else d["id"]
                out.append({"pipeline": p["id"], "column": d["id"], "key": key, "irds": irds, "slug": slug(irds),
                            "weight": p["weight"] * d["weight"] / len(d.get("subsets") or [1])})
    for p, build in spec.get("msmarco_build_weight", {}).items():   # index build cost, spread over MS MARCO cells
        ms = [c for c in out if c["pipeline"] == p and c["irds"].startswith("msmarco-passage")]
        for c in ms:
            c["weight"] += build / len(ms)
    return out


def run_path(run_dir: Path, cell: dict) -> Path:
    return run_dir / "runs" / cell["slug"] / f"{cell['pipeline']}.res.gz"


def find_run(run_dir: Path, cell: dict) -> Path | None:
    """The run file, also accepting an uncompressed .res or a .trec next to it."""
    base = run_path(run_dir, cell)
    for p in (base, base.with_suffix(""), base.with_name(f"{cell['pipeline']}.trec")):
        if p.exists():
            return p
    return None


def check_run(path: Path, cache: dict) -> dict:
    """Parse a TREC run once per (mtime, size): valid, number of queries and rows."""
    st = path.stat()
    key = str(path)
    if cache.get(key, {}).get("stamp") == [st.st_mtime, st.st_size]:
        return cache[key]
    qids, rows, ok = set(), 0, True
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    ok = False
                    break
                float(parts[4])
                qids.add(parts[0])
                rows += 1
    except (OSError, EOFError, ValueError):
        ok = False
    cache[key] = {"stamp": [st.st_mtime, st.st_size], "valid": ok and rows > 0, "queries": len(qids), "rows": rows,
                  "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    return cache[key]


def agent_status(run_dir: Path) -> dict:
    """status.json as {(pipeline, key): {"state", "updated", "note"}}; tolerant of plain-string states and slug keys."""
    try:
        raw = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for pipeline, per in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(per, dict):
            continue
        for key, v in per.items():
            v = v if isinstance(v, dict) else {"state": str(v)}
            out[(pipeline, normalise_key(key))] = v
    return out


def normalise_key(key: str) -> str:
    k = key.strip().lower()
    aliases = {"msmarco-passage/dev/small": "msmarco-dev", "msmarco-passage-dev-small": "msmarco-dev", "dev": "msmarco-dev",
               "msmarco-passage/trec-dl-2019/judged": "dl19", "msmarco-passage-trec-dl-2019-judged": "dl19",
               "msmarco-passage/trec-dl-2020/judged": "dl20", "msmarco-passage-trec-dl-2020-judged": "dl20"}
    if k in aliases:
        return aliases[k]
    if k.startswith("nano-beir-"):
        return "nano-beir/" + k[len("nano-beir-"):]
    return k


def cell_states(run_dir: Path, cells: list[dict], cache: dict, expected_queries: dict | None = None) -> dict:
    """(pipeline, key) -> state. A valid run file means done, whatever status.json says."""
    status = agent_status(run_dir)
    out = {}
    for c in cells:
        p = find_run(run_dir, c)
        info = check_run(p, cache) if p else None
        want = (expected_queries or {}).get(c["irds"])
        if info and info["valid"] and (not want or info["queries"] >= 0.9 * want):
            out[(c["pipeline"], c["key"])] = "done"
        else:
            s = status.get((c["pipeline"], c["key"]), {}).get("state", "pending")
            out[(c["pipeline"], c["key"])] = "pending" if s == "done" or s not in STATES else s   # claimed done, no valid file
    return out


def condition_cells(run_dir: Path, spec: dict) -> list[dict]:
    try:
        wanted = json.loads((run_dir / "cells.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return all_cells(spec)
    return [c for c in all_cells(spec) if [c["pipeline"], c["key"]] in wanted]


def condition_dirs(iteration: Path, spec: dict) -> list[Path]:
    order = [c["id"] for c in spec["conditions"]]
    return [iteration / c for c in order if (iteration / c / "cells.json").exists()]


def latest_iteration(workspace: Path = DEFAULT_WORKSPACE) -> Path | None:
    its = [p for p in workspace.glob("pipelines-*") if p.is_dir() and p.name.split("-")[1].isdigit()]
    return max(its, key=lambda p: int(p.name.split("-")[1])) if its else None


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def status_log(run_dir: Path) -> list[dict]:
    path = run_dir / "status_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cell_times(run_dir: Path) -> dict:
    """(pipeline, key) -> {"first_active", "done"} ISO times, from the runner's status log."""
    out = {}
    for e in status_log(run_dir):
        k = (e["pipeline"], e["key"])
        t = out.setdefault(k, {})
        if e["state"] in ("building", "running") and "first_active" not in t:
            t["first_active"] = e["time"]
        if e["state"] == "done" and "done" not in t:
            t["done"] = e["time"]
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def expected_queries(spec: dict) -> dict:
    return {d["irds"]: d["queries"] for d in spec["datasets"] if d.get("queries")}
