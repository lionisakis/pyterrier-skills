#!/usr/bin/env python3
"""Run the pipeline-matrix evals: one agent per resource condition builds 9 pipelines on 4 dataset groups.

The conditions turn three resources on and off: the skills (copied into `.claude/skills/`), the
documentation (the `pyterrier-docs/` mirror from fetch_docs.py, search_docs.py and web access) and code
view (reading the installed packages' source). evals/pipelines/block_source.py enforces the rules.
The pipelines, datasets and conditions are in evals/pipelines/pipelines.json; the prompt template is
evals/pipelines/agent_instructions.md.

Each condition gets `<workspace>/pipelines-N/<condition>/`. The agent creates its venv and indexes
there and writes runs to `runs/<slug>/<pipeline>.res.gz`. When it stops with cells missing, the runner
resumes the session with a fixed nudge (counted as a human message), up to --max-nudges times.
Before each attempt it wakes the model if the gateway reports it asleep, and while a condition runs it pings
the model every --keepalive-min minutes so it doesn't fall asleep while the agent waits on a long job.

Watch progress with `python evals/pipeline_status.py --watch 60`; grade with grade_pipelines.py; then
pipeline_report.py.

Examples
    python evals/run_pipelines.py --model qwen-3.8-27b --effort medium --dry-run
    python evals/run_pipelines.py --model qwen-3.8-27b --effort medium
    python evals/run_pipelines.py --model qwen-3.8-27b --effort medium --conditions none,code --iteration pipelines-1
    python evals/run_pipelines.py --model qwen-3.8-27b --effort medium --conditions all \\
        --pipelines bm25,bm25_prf_bm25 --datasets dl19          # smoke test
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipelines"))
from run_evals import (MARKER, REPO, SKILLS, TOOLS, CpuHeartbeat, KeepAlive, ProcTreeCpu, command,  # noqa: E402
                       gpu_status, hardware_summary, load_env_file, memory_limit_gb, merge_cpu, model_status,
                       ping_model, wait_for_model)
import matrix  # noqa: E402

PIPES = REPO / "evals" / "pipelines"
TEMPLATE = PIPES / "agent_instructions.md"
HOOK = PIPES / "block_source.py"
DOCS = REPO / "pyterrier-docs"
SEARCH_DOCS = REPO / "evals" / "search_docs.py"

RESOURCES = {
    ("skills", True): "- **Skills**: PyTerrier skills are installed in `.claude/skills/`. Open the ones whose "
                      "descriptions match the task and read SKILL.md first.",
    ("skills", False): "- **Skills**: none are installed.",
    ("docs", True): "- **Documentation**: a copy of the PyTerrier documentation (core and extensions) is in "
                    "`pyterrier-docs/`, one text file per page (`index.json` lists them). Search it with "
                    "`python3 search_docs.py \"<query>\"`. You may also use WebFetch/WebSearch if your tools support them.",
    ("docs", False): "- **Documentation**: none. There is no local copy and no web access; fetching documentation "
                     "sites is blocked.",
    ("code", True): "- **Code view**: you may read the source of everything installed in your venv (site-packages, "
                    "`inspect`, `help()`, `pip show -f`) and source repositories such as GitHub.",
    ("code", False): "- **Code view**: none. Reading installed packages' files (site-packages), `inspect`, `help()`, "
                     "`__file__`, `pip show -f` and source repositories are blocked. `pip install` works, and error "
                     "messages and tracebacks are fine to read.",
}
NUDGE = ("These cells are still missing (no valid run file in runs/<slug>/<pipeline>.res.gz): {missing}. "
         "Continue the task: finish them, or mark them failed in status.json with a reason, then update "
         "final_report.md.")


RETRY = "The last request failed with a server error. Continue the task where you left off."


INTERRUPTED = ("Your session was interrupted: the process was killed, and any background jobs you started were stopped "
               "with it. These cells are still missing (no valid run file in runs/<slug>/<pipeline>.res.gz): {missing}. "
               "Check what state your work is in, then continue the task: finish the missing cells (or mark them failed "
               "in status.json with a reason), rerun the evaluation and update final_report.md and manifest.json.")


def last_session(run_dir: Path) -> str | None:
    """The id of the condition's most recent claude session (its transcript file name), for --resume."""
    files = sorted((run_dir / "claude-config" / "projects").glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1].stem if files else None


def api_error(res: dict) -> bool:
    """A run that ended on a gateway/API error rather than because the agent stopped."""
    return bool(res.get("is_error")) and str(res.get("result", "")).startswith("API Error")


def next_batch(workspace: Path) -> Path:
    last = matrix.latest_iteration(workspace)
    return workspace / f"pipelines-{int(last.name.split('-')[1]) + 1 if last else 1}"


def render_prompt(spec: dict, cond: dict, cells: list[dict], run_dir: Path, base_python: str, hardware: str) -> str:
    pipes = {c["pipeline"] for c in cells}
    cols = {c["column"] for c in cells}
    pipelines = "\n".join(f"- `{p['id']}`: {p['spec']}" + (f" (models: {', '.join(p['models'])})" if p["models"] else "")
                          + (f" — only on {', '.join(f'`{d}`' for d in p['datasets'])}" if p.get("datasets") else "")
                          for p in spec["pipelines"] if p["id"] in pipes)
    datasets = []
    for d in spec["datasets"]:
        if d["id"] not in cols:
            continue
        if d.get("subsets"):
            datasets.append(f"- `nanobeir` — NanoBEIR, 13 sub-datasets: " + ", ".join(f"`{s}`" for s in d["subsets"])
                            + " (also on Hugging Face as zeta-alpha-ai/NanoArguAna, NanoClimateFEVER, ...)")
        else:
            datasets.append(f"- `{d['id']}` — `{d['irds']}` ({d['queries']} queries)")
    measures = ", ".join(sorted({m for d in spec["datasets"] if d["id"] in cols for m in d["metrics"]}))
    ms = [c for c in ("msmarco-dev", "dl19", "dl20") if c in cols]
    notes = ""
    if len(ms) > 1:
        notes += "These MS MARCO datasets share one corpus (`msmarco-passage`, 8.8M passages): build each index once and reuse it.\n"
    elif ms:
        notes += "The MS MARCO corpus (`msmarco-passage`) has 8.8M passages.\n"
    if "nanobeir" in cols:
        notes += "For NanoBEIR, every sub-dataset has its own small corpus and its own index; run every pipeline on all 13 sub-datasets.\n"
    status_ids = ", ".join([f"`{c}`" for c in ms] + (["`nano-beir/<name>` for each NanoBEIR subset"] if "nanobeir" in cols else []))
    example_slug = next(c["slug"] for c in cells)
    resources = "\n".join(RESOURCES[(k, cond[k])] for k in ("skills", "docs", "code"))
    return TEMPLATE.read_text(encoding="utf-8").format(
        workspace=run_dir, base_python=base_python, hardware=hardware, resources=resources,
        pipelines=pipelines, datasets="\n".join(datasets), measures=measures, prompt=spec["prompt"],
        dataset_notes=notes, status_ids=status_ids, example_slug=example_slug)


def prepare(spec: dict, cond: dict, cells: list[dict], iteration: Path, base_python: str, hardware: str) -> Path:
    run_dir = iteration / cond["id"]
    project = run_dir / "project"
    (project / ".claude").mkdir(parents=True, exist_ok=True)
    for sub in ("runs", "indexes", "scripts", "logs", "results"):
        (run_dir / sub).mkdir(exist_ok=True)
    if cond["skills"]:
        for skill in SKILLS.iterdir():
            if (skill / "SKILL.md").exists():
                shutil.copytree(skill, project / ".claude" / "skills" / skill.name, dirs_exist_ok=True)
    if cond["docs"]:
        if not (DOCS / "index.json").exists():
            raise SystemExit(f"{DOCS} is missing: run `python evals/fetch_docs.py` first")
        if not (project / "pyterrier-docs").exists():
            (project / "pyterrier-docs").symlink_to(DOCS, target_is_directory=True)
        shutil.copy(SEARCH_DOCS, project / "search_docs.py")
    base_real = os.path.realpath(base_python)
    hook_cfg = {"run_dir": str(run_dir.resolve()), "iteration": str(iteration.resolve()), "repo": str(REPO),
                "docs_dir": str(DOCS.resolve()), "docs": cond["docs"], "code": cond["code"],
                "allow": [os.path.dirname(os.path.dirname(base_real))] if base_real.startswith(str(REPO)) else []}
    (run_dir / "hook_config.json").write_text(json.dumps(hook_cfg, indent=2), encoding="utf-8")
    settings = {"hooks": {"PreToolUse": [{"matcher": "Read|Grep|Glob|LS|Bash|Write|Edit|MultiEdit|WebFetch|WebSearch",
                                          "hooks": [{"type": "command",
                                                     "command": f"python3 {HOOK} {run_dir.resolve() / 'hook_config.json'}"}]}]}}
    (project / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (run_dir / "cells.json").write_text(json.dumps([[c["pipeline"], c["key"]] for c in cells]), encoding="utf-8")
    (run_dir / "prompt.md").write_text(render_prompt(spec, cond, cells, run_dir.resolve(), base_python, hardware), encoding="utf-8")
    write_state(run_dir, state="queued")
    return run_dir


def write_state(run_dir: Path, **kw) -> None:
    state = matrix.read_json(run_dir / "agent_state.json", {}) or {}
    state.update(kw, updated=matrix.now_iso())
    (run_dir / "agent_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


class Monitor(threading.Thread):
    """Samples GPU use every 5 s, logs cell state changes every 30 s and prints a heartbeat every 60 s."""

    def __init__(self, run_dir: Path, cells: list[dict], spec: dict, tree: ProcTreeCpu | None = None):
        super().__init__(daemon=True)
        self.run_dir, self.cells, self.spec, self.tree = run_dir, cells, spec, tree
        self.stop = threading.Event()
        self.gpu_s = self.util_sum = 0.0
        self.samples = self.mem_peak = 0
        self.gpu_now: tuple[float, int, int] | None = None   # (util %, used MiB, total MiB) of the last sample
        self.ram_total_gb = memory_limit_gb()
        self.cpu_mark = (time.time(), 0.0)               # (wall time, tree CPU seconds) at the last heartbeat
        self.cache: dict = {}
        self.last: dict = {}

    def gpu_sample(self, dt: float) -> None:
        if not shutil.which("nvidia-smi"):
            return
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        rows = []   # skip lines such as "No devices were found" (nvidia-smi installed, no GPU)
        for line in out.strip().splitlines():
            try:
                rows.append([float(x) for x in line.split(",")[:3]])
            except ValueError:
                pass
        rows = [r for r in rows if len(r) == 3]
        if rows:
            self.gpu_s += sum(u for u, _, _ in rows) / 100 * dt
            self.util_sum += sum(u for u, _, _ in rows) / len(rows)
            self.samples += 1
            self.mem_peak = max(self.mem_peak, int(sum(m for _, m, _ in rows)))
            self.gpu_now = (sum(u for u, _, _ in rows) / len(rows), int(sum(m for _, m, _ in rows)),
                            int(sum(t for _, _, t in rows)))

    def log_states(self) -> None:
        states = matrix.cell_states(self.run_dir, self.cells, self.cache, matrix.expected_queries(self.spec))
        with (self.run_dir / "status_log.jsonl").open("a", encoding="utf-8") as log:
            for k, s in states.items():
                if self.last.get(k) != s:
                    log.write(json.dumps({"time": matrix.now_iso(), "pipeline": k[0], "key": k[1], "state": s}) + "\n")
        self.last = states

    def heartbeat(self) -> None:
        # a launcher that kills silent jobs must not see the long model waits (first turn: ~10 min) as idle
        done = sum(s == "done" for s in self.last.values())
        line = (f"[{self.run_dir.name}] heartbeat {datetime.now():%H:%M:%S}  up {(time.time() - self.started) / 60:.0f} min"
                f"  {done}/{len(self.cells)} cells done")
        if self.tree:   # CPU of the agent's process tree since the last heartbeat; 100% = one core
            now, cpu = time.time(), sum(u + s for u, s in self.tree.cpu.values())
            line += f"  cpu {100 * (cpu - self.cpu_mark[1]) / max(now - self.cpu_mark[0], 1e-9):.0f}%  ram {self.tree.rss / 2**30:.1f}/{self.ram_total_gb:.0f} GB"
            self.cpu_mark = (now, cpu)
        line += (f"  gpu {self.gpu_now[0]:.0f}%  vram {self.gpu_now[1] / 1024:.1f}/{self.gpu_now[2] / 1024:.0f} GB"
                 if self.gpu_now else "  gpu n/a")
        print(line, flush=True)

    def run(self) -> None:
        tick, self.started = 0, time.time()
        while not self.stop.wait(5):
            tick += 1
            for step in (lambda: self.gpu_sample(5), self.log_states if tick % 6 == 0 else None,
                         self.heartbeat if tick % 12 == 0 else None):
                try:   # one bad sample must not kill the monitor (and with it the cell timings)
                    step and step()
                except Exception as e:  # noqa: BLE001
                    print(f"[monitor] {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def result(self) -> dict:
        return {"gpu_seconds": round(self.gpu_s), "gpu_util_mean": round(self.util_sum / self.samples, 1) if self.samples else None,
                "gpu_mem_peak_mb": self.mem_peak or None}


def merge_results(results: list[dict]) -> dict:
    """One result.json for report.eval_report: usage, API time, turns and cost summed over attempts."""
    merged = dict(results[-1]) if results else {}
    usage: dict = {}
    for r in results:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    merged.update(usage=usage,
                  duration_api_ms=sum(r.get("duration_api_ms") or 0 for r in results),
                  duration_ms=sum(r.get("duration_ms") or 0 for r in results),
                  num_turns=sum(r.get("num_turns") or 0 for r in results),
                  total_cost_usd=sum(r.get("total_cost_usd") or 0 for r in results),
                  subtypes=[r.get("subtype") for r in results])
    return merged


def run_condition(spec: dict, cond: dict, cells: list[dict], iteration: Path, args, env: dict[str, str]) -> dict:
    run_dir = iteration / cond["id"]
    project = run_dir / "project"
    tools = TOOLS + (",WebFetch,WebSearch" if cond["docs"] else "")
    cmd = command({}, run_dir, args.model, args.max_turns, args.effort)
    cmd[cmd.index("--allowedTools") + 1] = tools
    # a private PyTerrier home: ~/.pyterrier is a symlink into NFS on these nodes, and a stale mount makes
    # pt.io.pyterrier_home() fail with FileExistsError for every Terrier call
    env = {"PYTERRIER_HOME": str(run_dir / "pyterrier_home"), **env, "BASH_MAX_TIMEOUT_MS": "600000",
           MARKER: str(run_dir)}   # lets ProcTreeCpu count background (nohup) jobs too
    if "ANTHROPIC_API_KEY" in env:
        env["CLAUDE_CONFIG_DIR"] = str(run_dir / "claude-config")
        (run_dir / "claude-config").mkdir(exist_ok=True)
    if args.dry_run:
        print(f"[{cond['id']}] (cd {project} && {' '.join(cmd)} < {run_dir / 'prompt.md'})")
        return {"condition": cond["id"], "status": "dry-run"}

    tree = ProcTreeCpu(str(run_dir), activity_log=run_dir / "activity.jsonl")
    tree.start()
    monitor = Monitor(run_dir, cells, spec, tree)
    monitor.start()
    keepalive = KeepAlive(env, args.model, args.keepalive_min, cond["id"]) if args.keepalive_min > 0 else None
    keepalive and keepalive.start()
    log = lambda m: print(f"[{cond['id']}] {m}", flush=True)  # noqa: E731
    started = time.time()
    results, exits, cpu_u, cpu_s, rss, nudges, api_retries = [], [], 0.0, 0.0, 0, 0, 0
    prompt = stdin_text = (run_dir / "prompt.md").read_text(encoding="utf-8")
    attempt_cmd, first, resumed = cmd, 0, None
    if args.resume:   # continue the killed session instead of starting over; keep its earlier state and result files
        resumed = {"session": last_session(run_dir), "at": matrix.now_iso(), "host": socket.gethostname(),
                   "previous": matrix.read_json(run_dir / "agent_state.json", {}) or {}}
        if not resumed["session"] or not resumed["previous"].get("started"):
            sys.exit(f"[{cond['id']}] --resume: no earlier session in {run_dir / 'claude-config'}")
        resumed["killed"] = max(p.stat().st_mtime for p in (run_dir / "claude-config" / "projects").glob(f"*/{resumed['session']}.jsonl"))
        first = len(list(run_dir.glob("result-*.json")))
        monitor.log_states()
        missing = [f"{p}/{k}" for (p, k), s in monitor.last.items() if s not in ("done", "failed")]
        stdin_text = INTERRUPTED.format(missing=", ".join(missing[:60]) + (f" and {len(missing) - 60} more" if len(missing) > 60 else ""))
        attempt_cmd = cmd + ["--resume", resumed["session"]]
        log(f"resuming session {resumed['session']}: {len(missing)} cells missing")
        write_state(run_dir, state="running", host=socket.gethostname(), pid=os.getpid(),
                    resumed=resumed["previous"].get("resumed", []) + [{k: resumed[k] for k in ("session", "at", "host")}])
    else:
        write_state(run_dir, state="running", started=matrix.now_iso(), host=socket.gethostname(), pid=os.getpid())
    while True:
        attempt = first + len(results)
        out_path = run_dir / f"result-{attempt}.json"
        if model_status(env, args.model) not in (None, "running"):
            write_state(run_dir, state="waking model")
            wait_for_model(env, args.model, args.wake_timeout_min * 60, log)
            write_state(run_dir, state="running")
        with out_path.open("w") as out, (run_dir / "stderr.txt").open("a") as err:
            proc = subprocess.Popen(attempt_cmd, cwd=project, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                    env={**os.environ, **env}, text=True)
            proc.stdin.write(stdin_text)
            proc.stdin.close()
            _, status, rusage = os.wait4(proc.pid, 0)
        exits.append(os.waitstatus_to_exitcode(status))
        cpu_u, cpu_s, rss = cpu_u + rusage.ru_utime, cpu_s + rusage.ru_stime, max(rss, rusage.ru_maxrss)
        res = matrix.read_json(out_path, {}) or {}
        results.append(res)
        monitor.log_states()
        missing = [f"{p}/{k}" for (p, k), s in monitor.last.items() if s not in ("done", "failed")]
        write_state(run_dir, attempts=len(results), exits=exits, nudges=nudges, api_retries=api_retries, missing=len(missing))
        if api_error(res) and api_retries < args.api_retries:
            # a gateway error (e.g. 503 "model is starting up") is not the agent stopping: wait and retry, no nudge;
            # also with no cells missing, since the agent may still have to evaluate and write its report
            api_retries += 1
            # a model the gateway reports as not running is waited on before the next attempt instead
            wait = min(60 * api_retries, 600) if model_status(env, args.model) in (None, "running") else 0
            print(f"[{cond['id']}] API error, retry {api_retries}/{args.api_retries} in {wait}s: {str(res.get('result'))[:100]}", flush=True)
            write_state(run_dir, state="waiting for API", api_retries=api_retries)
            time.sleep(wait)
            write_state(run_dir, state="running")
            if (res.get("num_turns") or 0) <= 1 and not resumed:   # nothing happened yet: start again with the prompt
                stdin_text, attempt_cmd = prompt, cmd
            elif (res.get("num_turns") or 0) <= 1:   # the resume itself failed: send the same message again
                attempt_cmd = cmd + ["--resume", res.get("session_id") or resumed["session"]]
            else:
                stdin_text, attempt_cmd = RETRY, cmd + ["--resume", res["session_id"]]
            continue
        if not missing or nudges >= args.max_nudges or not res.get("session_id"):
            break
        nudges += 1
        stdin_text = NUDGE.format(missing=", ".join(missing[:60]) + (f" and {len(missing) - 60} more" if len(missing) > 60 else ""))
        attempt_cmd = cmd + ["--resume", res["session_id"]]
        print(f"[{cond['id']}] nudge {nudges}: {len(missing)} cells missing", flush=True)
    monitor.stop.set()
    monitor.join()
    if keepalive:
        keepalive.stop.set()
    elapsed = time.time() - started

    merged = merge_results(results)
    (run_dir / "result.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (run_dir / "final_message.md").write_text(str(merged.get("result", "")), encoding="utf-8")
    if resumed:   # the killed run's time (its start to its transcript's last write) counts too; the gap does not
        resumed["previous_seconds"] = round(resumed["killed"] - datetime.fromisoformat(resumed["previous"]["started"]).timestamp(), 1)
        elapsed += resumed["previous_seconds"]
    timing = {"total_duration_seconds": round(elapsed, 1), **merge_cpu(cpu_u, cpu_s, rss, tree.finish()),
              "resumed": {k: resumed[k] for k in ("session", "at", "host", "previous_seconds")} if resumed else None,
              "gpu": gpu_status(), **monitor.result(), "nudges": nudges,
              "api_retries": api_retries, "keepalive_pings": keepalive.pings if keepalive else 0,
              "attempts": len(results), "exits": exits}
    for k in ("total_cost_usd", "duration_ms", "num_turns", "usage"):
        timing[k] = merged.get(k)
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    done = sum(s == "done" for s in monitor.last.values())
    outcome = ("done" if done == len(cells) else "max-turns" if "error_max_turns" in merged.get("subtypes", [])
               else "out of nudges" if nudges >= args.max_nudges else "crashed" if exits[-1] != 0 else "stopped")
    write_state(run_dir, state="finished", finished=matrix.now_iso(), outcome=outcome, cells_done=done, cells=len(cells))
    print(f"[{cond['id']}] {outcome}: {done}/{len(cells)} cells in {elapsed / 3600:.1f} h, {nudges} nudges", flush=True)
    return {"condition": cond["id"], "status": outcome, "cells_done": done, "cells": len(cells), "seconds": round(elapsed)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen-3.8-27b", help="model for `claude --model`")
    ap.add_argument("--effort", default=None, help="`claude --effort` (the Glasgow gateway accepts low, medium or xhigh)")
    ap.add_argument("--conditions", help="comma-separated condition ids (default: all 8)")
    ap.add_argument("--pipelines", help="comma-separated pipeline ids (default: all 9)")
    ap.add_argument("--datasets", help="comma-separated dataset ids: msmarco-dev, dl19, dl20, nanobeir (default: all)")
    ap.add_argument("--jobs", type=int, default=1, help="conditions run at once (1 keeps time and GPU numbers clean)")
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--max-nudges", type=int, default=3)
    ap.add_argument("--api-retries", type=int, default=10,
                    help="retries after a gateway error such as 503 'model is starting up' (backoff 1-10 min; not nudges)")
    ap.add_argument("--keepalive-min", type=float, default=10,
                    help="send a 1-token request this often while a condition runs, so the gateway keeps the model "
                         "awake while the agent waits on long jobs (0 = off)")
    ap.add_argument("--wake-timeout-min", type=float, default=45,
                    help="before each attempt, wait up to this long for a sleeping model to start")
    ap.add_argument("--resume", action="store_true",
                    help="continue each condition's last claude session after it was killed (needs --iteration), "
                         "telling the agent which cells are missing; keeps its files and adds its earlier time")
    ap.add_argument("--cpu-heartbeat-s", type=float, default=10,
                    help="keep one core busy this many seconds per minute, so a pod culler that treats low CPU as idle "
                         "doesn't kill the batch (default 10 ≈ 0.17 cores; 0 = off; max 50)")
    ap.add_argument("--base-python", default=shutil.which("python3") or "python3",
                    help="Python the agent creates its venv from (needs 3.10+; Java 11+ must be on the node for Terrier)")
    ap.add_argument("--env-file", type=Path, default=None, help="KEY=VALUE file for each `claude` (default: .env in the repo or its parent)")
    ap.add_argument("--workspace", type=Path, default=matrix.DEFAULT_WORKSPACE, help="where pipelines-N/ folders go (use node-local disk)")
    ap.add_argument("--node", help="the cluster's name for this node size, e.g. a6000-xxl; stated in the prompt")
    ap.add_argument("--hardware", help="override the detected hardware line in the prompt with this text")
    ap.add_argument("--iteration", help="existing or new pipelines-N folder name (to split one batch across nodes)")
    ap.add_argument("--list", action="store_true", help="print pipelines, datasets and conditions, then exit")
    ap.add_argument("--dry-run", action="store_true", help="prepare folders and print commands, run nothing")
    args = ap.parse_args()
    args.cpu_heartbeat_s = min(max(args.cpu_heartbeat_s, 0), 50)
    if args.resume and not args.iteration:
        ap.error("--resume needs --iteration (the batch to continue)")

    spec = matrix.load_spec()
    if args.list:
        for key in ("pipelines", "datasets", "conditions"):
            print(f"{key}: {', '.join(x['id'] for x in spec[key])}")
        return 0
    split = lambda s: [x.strip() for x in s.split(",")] if s else None  # noqa: E731
    conds = [c for c in spec["conditions"] if not args.conditions or c["id"] in split(args.conditions)]
    cells = matrix.all_cells(spec, split(args.pipelines), split(args.datasets))
    if not conds or not cells:
        print("no conditions or no cells selected", file=sys.stderr)
        return 2
    if not args.dry_run and not shutil.which("claude"):
        print("`claude` CLI not on PATH", file=sys.stderr)
        return 2

    env_file = args.env_file or next((p for p in (REPO / ".env", REPO.parent / ".env") if p.exists()), None)
    env = load_env_file(env_file)
    if env_file:
        print(f"env: {env_file} ({', '.join(sorted(env))})", flush=True)

    iteration = args.workspace / args.iteration if args.iteration else next_batch(args.workspace)
    iteration.mkdir(parents=True, exist_ok=True)
    meta = matrix.read_json(iteration / "run_meta.json", {}) or {}
    meta.update(model=args.model, effort=args.effort, max_turns=args.max_turns, max_nudges=args.max_nudges,
                started=meta.get("started") or f"{datetime.now():%Y-%m-%d %H:%M}", gpu=gpu_status())
    meta.setdefault("nodes", []).append({"host": socket.gethostname(), "conditions": [c["id"] for c in conds],
                                         "jobs": args.jobs, "started": matrix.now_iso(), "gpu": gpu_status()})
    meta["jobs"] = max(n["jobs"] for n in meta["nodes"])
    (iteration / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"batch: {iteration}  {len(conds)} conditions × {len(cells)} cells  model {args.model}", flush=True)

    hardware = args.hardware or hardware_summary(iteration, args.node)
    print(f"hardware: {hardware}", flush=True)
    meta["nodes"][-1]["hardware"] = hardware
    (iteration / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for cond in conds:
        if args.resume:   # keep the existing prompt, hook config and state; the folder must already be there
            if not (iteration / cond["id"] / "prompt.md").exists():
                print(f"--resume: {iteration / cond['id']} has no earlier run", file=sys.stderr)
                return 2
            continue
        prepare(spec, cond, cells, iteration, args.base_python, hardware)
    busy = CpuHeartbeat(args.cpu_heartbeat_s) if args.cpu_heartbeat_s > 0 and not args.dry_run else None
    busy and busy.start()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda c: run_condition(spec, c, cells, iteration, args, env), conds))
    if busy:
        busy.stop.set()
    (iteration / f"summary-{socket.gethostname()}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["status"] in ("done", "dry-run") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
