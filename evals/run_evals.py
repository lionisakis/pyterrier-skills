#!/usr/bin/env python3
"""Run the task prompts in evals.json through `claude -p` with the skills installed, and collect findings.

Each eval gets a fresh project directory (outside this repo, so the copied SKILL.md files don't trip
`test_no_stray_skill_files`) with every skill copied into `.claude/skills/`. The agent does the task
there and writes `findings.md`. Read the findings, turn each defect into a "Don't" bullet or a
Troubleshooting row (see CLAUDE.md), re-run `pytest`, then re-run the eval to confirm.

Examples
    python evals/run_evals.py --list
    python evals/run_evals.py --id 3                 # one eval, default model
    python evals/run_evals.py --model opus --jobs 3  # all evals, three at a time
    python evals/run_evals.py --dry-run              # print the commands only
    python evals/run_evals.py --id 6 --model qwen-3.8-27b --effort medium   # endpoint from ../.env

While an eval runs, each one prints a heartbeat line a minute and (with --model) pings the model every
--keepalive-min minutes so the gateway doesn't put it to sleep; --cpu-heartbeat-s keeps a pod culler
that treats low CPU as idle from killing the batch.

`--env-file` (default: the first `.env` found in this repo or its parent) is read as KEY=VALUE lines
and passed to each `claude` process, overriding the current environment. Use it for
ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY and CLAUDE_CODE_AUTO_COMPACT_WINDOW. When it sets
ANTHROPIC_API_KEY, each run also gets an empty CLAUDE_CONFIG_DIR: otherwise the CLI sends the saved
claude.ai login instead of the key. Gateways that reject the default effort need `--effort`
(the Glasgow gateway accepts low, medium or xhigh).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "skills"
EVALS = REPO / "evals" / "evals.json"
TEMPLATE = REPO / "evals" / "agent_instructions.md"
DEFAULT_WORKSPACE = REPO.parent / f"{REPO.name}-evals"
PYTHON = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", REPO / ".venv")).resolve() / "bin" / "python"
TOOLS = "Bash,Read,Write,Edit,Glob,Grep"


def load_env_file(path: Path | None) -> dict[str, str]:
    """KEY=VALUE lines (optionally prefixed with `export`); blank lines and # comments are skipped."""
    if path is None:
        return {}
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def gateway_request(env: dict[str, str], path: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    """GET (or POST `body` to) `<ANTHROPIC_BASE_URL>/v1<path>`; returns (HTTP status, JSON), status 0 if unreachable."""
    base = env.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    if not base:
        return 0, {}
    key = env.get("ANTHROPIC_API_KEY", "")
    req = urllib.request.Request(base + ("" if base.endswith("/v1") else "/v1") + path,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "x-api-key": key,
                                          "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except (OSError, ValueError):
        return 0, {}


def model_status(env: dict[str, str], model: str) -> str | None:
    """The gateway's `ida_llm_api.status` for `model` (sleeping, running, ...); None if the gateway doesn't report one."""
    _, data = gateway_request(env, "/models")
    for m in data.get("data") or []:
        if m.get("id") == model and isinstance(m.get("ida_llm_api"), dict):
            return m["ida_llm_api"].get("status")
    return None


def ping_model(env: dict[str, str], model: str) -> int:
    """A 1-token request: wakes a sleeping model and resets its idle timer. Returns the HTTP status."""
    return gateway_request(env, "/messages", {"model": model, "max_tokens": 1,
                                              "messages": [{"role": "user", "content": "ping"}]}, timeout=60)[0]


def wait_for_model(env: dict[str, str], model: str | None, timeout_s: float, log=print) -> bool:
    """Wake `model` if the gateway says it is not running and wait until it is; True once it is (or can't tell)."""
    if not model or (status := model_status(env, model)) in (None, "running"):
        return True
    log(f"waking {model} (status {status})")
    started = last_ping = time.time()
    ping_model(env, model)
    while time.time() - started < timeout_s:
        time.sleep(30)
        if (status := model_status(env, model)) == "running":
            log(f"{model} ready after {time.time() - started:.0f}s")
            return True
        if time.time() - last_ping > 300:   # the first request may have timed out before a job was queued
            ping_model(env, model)
            last_ping = time.time()
    log(f"{model} still {status} after {timeout_s:.0f}s; going ahead")
    return False


MARKER = "EVAL_RUN_MARKER"


class ProcTreeCpu(threading.Thread):
    """CPU time and peak memory of every process carrying `EVAL_RUN_MARKER=<marker>` in its environment.

    wait4 only counts processes the agent waited for; jobs it starts with `nohup ... &` are reparented
    and escape it. Children inherit the environment, so this finds them too. Sampled every `period` s:
    a process that ends between samples loses at most that much CPU (wait4 still covers waited ones).
    """

    def __init__(self, marker: str, period: float = 5.0):
        super().__init__(daemon=True)
        self.needle = f"{MARKER}={marker}\0".encode()
        self.period, self.stop = period, threading.Event()
        self.cpu: dict = {}     # (pid, starttime) -> (utime_s, stime_s)
        self.peak_rss = self.rss = 0
        self.tick, self.page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")

    def sample(self) -> None:
        rss = 0
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    if self.needle not in f.read() + b"\0":
                        continue
                with open(f"/proc/{pid}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            # fields after the command: state is [0], utime [11], stime [12], starttime [19], rss pages [21]
            self.cpu[(pid, fields[19])] = (int(fields[11]) / self.tick, int(fields[12]) / self.tick)
            rss += int(fields[21]) * self.page
        self.rss, self.peak_rss = rss, max(self.peak_rss, rss)

    def run(self) -> None:
        while not self.stop.wait(self.period):
            try:
                self.sample()
            except Exception:  # noqa: BLE001  never let accounting kill a run
                pass

    def finish(self) -> dict:
        self.stop.set()
        self.join()
        return {"tree_cpu_user_s": round(sum(u for u, _ in self.cpu.values()), 1),
                "tree_cpu_system_s": round(sum(s for _, s in self.cpu.values()), 1),
                "tree_peak_rss_mb": round(self.peak_rss / 2**20), "tree_processes": len(self.cpu)}


def merge_cpu(cpu_u: float, cpu_s: float, rss_kb: int, tree: dict) -> dict:
    """The larger of wait4's numbers and the process-tree sampler's, per measure."""
    return {"cpu_user_s": round(max(cpu_u, tree["tree_cpu_user_s"]), 1),
            "cpu_system_s": round(max(cpu_s, tree["tree_cpu_system_s"]), 1),
            "max_rss_mb": max(round(rss_kb / 1024), tree["tree_peak_rss_mb"]), **tree}


class CpuHeartbeat(threading.Thread):
    """Keeps one core busy for `busy_s` seconds every minute, so a pod culler that treats low CPU as idle doesn't
    kill the batch while the agent waits on the model (pipelines-4 `none` died after ~10 quiet minutes).
    The spin runs in a child without EVAL_RUN_MARKER, so it is not counted as the agent's CPU."""

    def __init__(self, busy_s: float):
        super().__init__(daemon=True)
        self.busy_s = busy_s
        self.stop = threading.Event()

    def run(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != MARKER}
        spin = f"import time\nend = time.time() + {self.busy_s}\nwhile time.time() < end: pass"
        while not self.stop.wait(60 - self.busy_s):
            try:
                subprocess.run([sys.executable, "-c", spin], env=env, timeout=self.busy_s + 30)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"[cpu-heartbeat] {type(e).__name__}: {e}", file=sys.stderr, flush=True)


class KeepAlive(threading.Thread):
    """Sends a 1-token request every `minutes` so the gateway doesn't put the model to sleep while the agent
    waits on a long background job (the next turn would otherwise hit "model is starting up")."""

    def __init__(self, env: dict[str, str], model: str, minutes: float, label: str):
        super().__init__(daemon=True)
        self.env, self.model, self.interval, self.label = env, model, minutes * 60, label
        self.stop = threading.Event()
        self.pings = 0

    def run(self) -> None:
        while not self.stop.wait(self.interval):
            status = ping_model(self.env, self.model)
            self.pings += 1
            if not 200 <= status < 300:
                print(f"[{self.label}] keep-alive: HTTP {status}", flush=True)


class Heartbeat(threading.Thread):
    """Prints one line a minute while an eval runs (up time, the agent's CPU and RAM, GPU), so a quiet
    eval isn't mistaken for a hung one and a launcher that kills silent jobs leaves it alone."""

    def __init__(self, label: str, tree: ProcTreeCpu, period: float = 60):
        super().__init__(daemon=True)
        self.label, self.tree, self.period = label, tree, period
        self.stop = threading.Event()
        self.ram_total_gb = memory_limit_gb()

    def line(self, up_s: float, cpu_pct: float) -> str:
        line = (f"[{self.label}] heartbeat {datetime.now():%H:%M:%S}  up {up_s / 60:.0f} min"
                f"  cpu {cpu_pct:.0f}%  ram {self.tree.rss / 2**30:.1f}/{self.ram_total_gb:.0f} GB")
        gpu = gpu_usage()
        return line + (f"  gpu {gpu[0]:.0f}%  vram {gpu[1] / 1024:.1f}/{gpu[2] / 1024:.0f} GB" if gpu else "  gpu n/a")

    def run(self) -> None:
        started = mark_t = time.time()
        mark_cpu = 0.0
        while not self.stop.wait(self.period):
            try:   # CPU of the agent's process tree since the last line; 100% = one core
                now, cpu = time.time(), sum(u + s for u, s in self.tree.cpu.values())
                print(self.line(now - started, 100 * (cpu - mark_cpu) / max(now - mark_t, 1e-9)), flush=True)
                mark_t, mark_cpu = now, cpu
            except Exception as ex:  # noqa: BLE001  never let the heartbeat kill a run
                print(f"[{self.label}] heartbeat {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_limit() -> float:
    """CPUs this container may use: the cgroup quota (v2 or v1), else the visible CPU count."""
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2 and not v2.startswith("max"):
        quota, period = v2.split()[:2]
        return int(quota) / int(period)
    quota, period = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return float(len(os.sched_getaffinity(0)))


def memory_limit_gb() -> float:
    """RAM this container may use: the cgroup limit (v2 or v1), else the machine's total."""
    total = next((int(l.split()[1]) * 1024 for l in (_read("/proc/meminfo") or "").splitlines()
                  if l.startswith("MemTotal")), 0)
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read(path)
        if v and v.isdigit() and 0 < int(v) < total:
            return int(v) / 2**30
    return total / 2**30


def gpu_details() -> list[str]:
    """One "name, memory" line per visible GPU."""
    if gpu_status() == "none":
        return []
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if "," in l]


def hardware_summary(path: Path, node: str | None = None) -> str:
    """What the agent can use, for the prompt: node type, GPUs, CPU and RAM limits, free disk."""
    gpus = gpu_details()
    free = shutil.disk_usage(path).free / 2**30
    parts = [f"node type {node}"] if node else []
    parts += [f"{len(gpus)} GPU ({'; '.join(gpus)})" if gpus else "no GPU (CPU only)",
              f"{cpu_limit():g} CPU cores", f"{memory_limit_gb():.0f} GB RAM", f"{free:.0f} GB free disk in the workspace"]
    return ", ".join(parts)


def gpu_status() -> str:
    """GPUs visible to the agents, or 'none'."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1") or os.environ.get("NVIDIA_VISIBLE_DEVICES") in ("none", "void"):
        return "none"
    if not shutil.which("nvidia-smi"):
        return "none"
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    return "; ".join(l for l in out.splitlines() if l.startswith("GPU")) or "none"   # not "No devices were found"


def gpu_usage() -> tuple[float, int, int] | None:
    """(mean util %, used MiB, total MiB) over the visible GPUs, or None."""
    if gpu_status() == "none":
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        try:
            rows.append([float(x) for x in line.split(",")[:3]])
        except ValueError:
            pass
    rows = [r for r in rows if len(r) == 3]
    if not rows:
        return None
    return sum(r[0] for r in rows) / len(rows), int(sum(r[1] for r in rows)), int(sum(r[2] for r in rows))


def load_evals() -> list[dict]:
    return json.loads(EVALS.read_text(encoding="utf-8"))["evals"]


def next_iteration(workspace: Path) -> Path:
    existing = sorted(int(p.name.split("-")[1]) for p in workspace.glob("iteration-*") if p.is_dir())
    return workspace / f"iteration-{(existing[-1] + 1) if existing else 1}"


def prepare(run_dir: Path, ev: dict, max_turns: int) -> Path:
    """Create the project dir with skills installed and the prompt file. Returns the prompt path."""
    project = run_dir / "project"
    skills_dst = project / ".claude" / "skills"
    skills_dst.mkdir(parents=True, exist_ok=True)
    for skill in SKILLS.iterdir():
        if (skill / "SKILL.md").exists():
            shutil.copytree(skill, skills_dst / skill.name, dirs_exist_ok=True)
    prompt = TEMPLATE.read_text(encoding="utf-8").format(
        python=PYTHON, workspace=run_dir, prompt=ev["prompt"]
    )
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    json.dump(
        {"eval_id": ev["id"], "eval_name": ev["name"], "prompt": ev["prompt"],
         "assertions": ev.get("expectations", []), "max_turns": max_turns},
        (run_dir / "eval_metadata.json").open("w", encoding="utf-8"), indent=2,
    )
    return run_dir / "prompt.md"


def command(ev: dict, run_dir: Path, model: str | None, max_turns: int, effort: str | None) -> list[str]:
    cmd = ["claude", "-p", "--output-format", "json", "--allowedTools", TOOLS,
           "--add-dir", str(run_dir), "--max-turns", str(max_turns)]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def run_one(ev: dict, iteration: Path, model: str | None, max_turns: int, effort: str | None,
            dry_run: bool, env: dict[str, str], api_retries: int = 10, keepalive_min: float = 0) -> dict:
    run_dir = iteration / f"eval-{ev['id']}-{ev['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    max_turns = ev.get("max_turns", max_turns)   # an eval may need more turns than the batch default
    prompt_path = prepare(run_dir, ev, max_turns)
    cmd = command(ev, run_dir, model, max_turns, effort)
    # a private PyTerrier home: ~/.pyterrier is a symlink into NFS here, and a stale mount breaks every Terrier call
    env = {"PYTERRIER_HOME": str(run_dir / "pyterrier_home"), **env, MARKER: str(run_dir)}
    if "ANTHROPIC_API_KEY" in env:   # an empty config dir, so the key is used instead of a saved login
        env = {**env, "CLAUDE_CONFIG_DIR": str(run_dir / "claude-config")}
        (run_dir / "claude-config").mkdir(exist_ok=True)
    project = run_dir / "project"
    if dry_run:
        print(f"[{ev['name']}] (cd {project} && {' '.join(cmd)} < {prompt_path})")
        return {"name": ev["name"], "status": "dry-run"}
    started = time.time()
    tree = ProcTreeCpu(str(run_dir))
    tree.start()
    heartbeat = Heartbeat(ev["name"], tree)
    heartbeat.start()
    keepalive = KeepAlive(env, model, keepalive_min, ev["name"]) if model and keepalive_min > 0 else None
    keepalive and keepalive.start()
    attempt_cmd, stdin_text, retries, cpu_u, cpu_s, rss = cmd, prompt_path.read_text(encoding="utf-8"), 0, 0.0, 0.0, 0
    while True:
        wait_for_model(env, model, 45 * 60, lambda m: print(f"[{ev['name']}] {m}", flush=True))
        with (run_dir / "result.json").open("w") as out, (run_dir / "stderr.txt").open("a") as err:
            proc = subprocess.Popen(attempt_cmd, cwd=project, stdin=subprocess.PIPE, stdout=out, env={**os.environ, **env},
                                    stderr=err, text=True)
            proc.stdin.write(stdin_text)
            proc.stdin.close()
            # wait4 gives the CPU time and peak memory of the agent plus every process it ran and waited for
            _, status, rusage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        cpu_u, cpu_s, rss = cpu_u + rusage.ru_utime, cpu_s + rusage.ru_stime, max(rss, rusage.ru_maxrss)
        try:
            res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            res = {}
        # a gateway error (e.g. 503 "model is starting up") is not the agent finishing: wait and retry
        if not (res.get("is_error") and str(res.get("result", "")).startswith("API Error")) or retries >= api_retries:
            break
        retries += 1
        # a model the gateway reports as not running is waited on by wait_for_model at the top of the loop instead
        wait = min(60 * retries, 600) if model_status(env, model) in (None, "running") else 0
        print(f"[{ev['name']}] API error, retry {retries}/{api_retries} in {wait}s: {str(res.get('result'))[:100]}", flush=True)
        time.sleep(wait)
        if (res.get("num_turns") or 0) > 1 and res.get("session_id"):
            attempt_cmd = cmd + ["--resume", res["session_id"]]
            stdin_text = "The last request failed with a server error. Continue the task where you left off."
    heartbeat.stop.set()
    if keepalive:
        keepalive.stop.set()
    elapsed = time.time() - started
    usage = {}
    try:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        usage = {k: result.get(k) for k in ("total_cost_usd", "duration_ms", "num_turns", "usage")}
        (run_dir / "final_message.md").write_text(result.get("result", ""), encoding="utf-8")
    except (json.JSONDecodeError, FileNotFoundError):
        pass
    compute = {**merge_cpu(cpu_u, cpu_s, rss, tree.finish()), "gpu": gpu_status(), "api_retries": retries,
               "keepalive_pings": keepalive.pings if keepalive else 0}
    json.dump({"total_duration_seconds": round(elapsed, 1), **compute, **usage},
              (run_dir / "timing.json").open("w", encoding="utf-8"), indent=2)
    findings = run_dir / "findings.md"
    status = "ok" if proc.returncode == 0 and findings.exists() else f"exit {proc.returncode}"
    print(f"[{ev['name']}] {status} in {elapsed:.0f}s -> {findings if findings.exists() else run_dir}", flush=True)
    return {"name": ev["name"], "status": status, "seconds": round(elapsed), "dir": str(run_dir)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", type=int, action="append", help="eval id to run (repeatable); default all")
    ap.add_argument("--model", default=None, help="model for `claude --model` (e.g. opus, sonnet)")
    ap.add_argument("--jobs", type=int, default=1, help="evals to run concurrently")
    ap.add_argument("--max-turns", type=int, default=150,
                    help="turn limit per eval; an eval's own \"max_turns\" in evals.json overrides it")
    ap.add_argument("--api-retries", type=int, default=10,
                    help="retries after a gateway error such as 503 'model is starting up' (backoff 1-10 min)")
    ap.add_argument("--effort", default=None, help="`claude --effort` level (low, medium, high, xhigh, max)")
    ap.add_argument("--keepalive-min", type=float, default=10,
                    help="send a 1-token request this often while an eval runs, so the gateway keeps the model "
                         "awake while the agent waits on long jobs (0 = off; needs --model)")
    ap.add_argument("--cpu-heartbeat-s", type=float, default=10,
                    help="keep one core busy this many seconds per minute, so a pod culler that treats low CPU as idle "
                         "doesn't kill the batch (default 10 ≈ 0.17 cores; 0 = off; max 50)")
    ap.add_argument("--env-file", type=Path, default=None,
                    help="KEY=VALUE file passed to each `claude` process (default: .env in the repo or its parent)")
    ap.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                    help=f"where iteration-N/ folders go (default {DEFAULT_WORKSPACE})")
    ap.add_argument("--list", action="store_true", help="print the evals and exit")
    ap.add_argument("--dry-run", action="store_true", help="prepare dirs and print commands, run nothing")
    args = ap.parse_args()
    args.cpu_heartbeat_s = min(max(args.cpu_heartbeat_s, 0), 50)

    evals = load_evals()
    if args.list:
        for ev in evals:
            print(f"{ev['id']:>2}  {ev['name']:<26} {', '.join(ev['skills'])}")
        return 0
    if args.id:
        evals = [ev for ev in evals if ev["id"] in args.id]
        if not evals:
            print("no eval with that id", file=sys.stderr)
            return 2
    if not args.dry_run and not shutil.which("claude"):
        print("`claude` CLI not on PATH", file=sys.stderr)
        return 2

    env_file = args.env_file or next((p for p in (REPO / ".env", REPO.parent / ".env") if p.exists()), None)
    env = load_env_file(env_file)
    if env_file:
        print(f"env: {env_file} ({', '.join(sorted(env))})", flush=True)

    iteration = next_iteration(args.workspace)
    iteration.mkdir(parents=True)
    print(f"iteration: {iteration}  started {datetime.now():%Y-%m-%d %H:%M}", flush=True)
    batch_started = time.time()
    meta = {"model": args.model, "effort": args.effort, "jobs": args.jobs, "max_turns": args.max_turns,
            "max_turns_by_eval": {ev["name"]: ev["max_turns"] for ev in evals if "max_turns" in ev},
            "started": f"{datetime.now():%Y-%m-%d %H:%M}", "gpu": gpu_status()}
    busy = CpuHeartbeat(args.cpu_heartbeat_s) if args.cpu_heartbeat_s > 0 and not args.dry_run else None
    busy and busy.start()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(
            lambda ev: run_one(ev, iteration, args.model, args.max_turns, args.effort, args.dry_run, env,
                               args.api_retries, args.keepalive_min),
            evals))
    if busy:
        busy.stop.set()
    json.dump(results, (iteration / "summary.json").open("w", encoding="utf-8"), indent=2)
    if not args.dry_run:
        meta["wall_s"] = round(time.time() - batch_started)
        (iteration / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        from report import write_report
        print(f"report: {write_report(iteration)}", flush=True)
    return 0 if all(r["status"] in ("ok", "dry-run") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
