"""Split a condition's wall time into agentic coding, index building, pipeline running, setup, downloads and other.

Agents start long jobs with `nohup ... &` and poll them, so tool-call durations mostly measure polling and the
real work overlaps the model's turns. Every second of a window therefore gets one label, by priority:

    index > build_and_run > pipeline > download > setup > agentic coding (model generating) > other

so a model turn taken while an index builds in the background counts as index building (the build is the
critical path). `llm_total_s` keeps the model time including the overlap.

Two sources for what ran when:
- measured: `activity.jsonl`, the runner's 5 s samples of the agent's live processes (runs from 2026-10-21 on);
- reconstructed: foreground tool calls from the transcript, plus background jobs from their `nohup ... > log`
  launch to the log file's last write (an earlier launch to the same log ends at the next one). Approximate.

A script (or wrapper) that both builds indexes and runs pipelines is reported as build_and_run: neither the
transcript nor process samples show where its build ends, the agents' status.json states are used inconsistently
(three conditions never mark a cell `building`) and manifest.json build times are self-reported (one agent wrote
cumulative times), so none of them is used to split it.

Usage
    python evals/pipelines/time_split.py ../pyterrier-skills-evals/pipelines-4/skills
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matrix  # noqa: E402
from report import BUILDS, read_rows  # noqa: E402

CATEGORIES = ("agentic_coding", "index_build", "pipeline_run", "build_and_run", "setup", "download", "other")
# label ids in priority order (higher wins); "down" = between a killed session and its resume, and "limit" = waiting
# for a subscription usage limit to reset (unless a job ran meanwhile): neither is counted
DOWN, INDEX, MIXED, PIPELINE, DOWNLOAD, SETUP, LIMIT, LLM, OTHER, IDLE = 10, 9, 8, 7, 6, 5, 4, 3, 2, 0
NAME = {INDEX: "index_build", MIXED: "build_and_run", PIPELINE: "pipeline_run", DOWNLOAD: "download", SETUP: "setup",
        LLM: "agentic_coding", OTHER: "other", IDLE: "other"}

SETUP_RE = re.compile(r"\b(pip3?|uv)\s+(pip\s+)?install\b|\b-m\s+(venv|pip\s+install)\b|\buv\s+(venv|sync)\b|"
                      r"\bconda\s+(install|create)\b|\bapt(-get)?\s+install\b|\bvirtualenv\b")
DOWNLOAD_RE = re.compile(r"\bwget\b|\bcurl\b|hf_hub_download|snapshot_download|huggingface-cli|\bhf\s+download\b|"
                         r"\bgit\s+clone\b|\.download\(")
INDEX_RE = re.compile(BUILDS.pattern + r"|Indexer\(|build_index|faiss\.(Index|index_factory)|\.encode_corpus\(")
PIPELINE_RE = re.compile(r"pt\.Experiment\(|\.transform\(|write_results|\.search\(|\.bm25\(|Retriever\(|"
                         r"\.(rerank|retrieve)\(|get_topics\(|ir_measures\.|calc_aggregate")
INDEX_NAME = re.compile(r"index|build|encode|embed", re.I)
PIPELINE_NAME = re.compile(r"run|retriev|rerank|eval|pipeline|experiment|cell|monot5|bert|prf|smoke|colbert", re.I)
DOWNLOAD_NAME = re.compile(r"fetch|download|dl_", re.I)
SETUP_NAME = re.compile(r"install|setup_env|env_setup", re.I)
SCRIPT_REF = re.compile(r"(?<![\w/.-])((?:[\w./-]*/)?[\w.-]+\.(?:py|sh))\b")
# `[nohup] <command> > <log> [2>&1] &` on one line (not `&&`)
BG_LAUNCH = re.compile(r"(?:nohup\s+|setsid\s+)?([^;&|\n]+?)\s*>>?\s*(\S+)(?:\s+2>&1)?\s*&(?!&)")
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?.*?\n.*?\n\1\b", re.S)
ASSIGN = re.compile(r"(?:^|[\s;&(])(?:export\s+)?([A-Za-z_]\w*)=(\"[^\"]*\"|'[^']*'|[^\s;&|]+)")


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Classifier:
    """Command or process text -> label, reading the scripts it runs from the condition's folder."""

    def __init__(self, run_dir: Path):
        self.run_dir, self.cache = run_dir, {}
        env = run_dir / "env.sh"   # agents often keep $PY, $WS, ... there and source it
        self.env = dict((k, v.strip("'\"")) for k, v in ASSIGN.findall(env.read_text(errors="replace"))) if env.is_file() else {}

    def expand(self, cmd: str) -> str:
        """Substitute $VAR / ${VAR} from assignments in the command and env.sh (one level)."""
        env = {**self.env, **{k: v.strip("'\"") for k, v in ASSIGN.findall(cmd)}}
        return re.sub(r"\$\{?([A-Za-z_]\w*)\}?", lambda m: env.get(m.group(1), m.group(0)), cmd)

    def path(self, ref: str, sub: str) -> Path | None:
        """A file the agent named: as given, relative to its folder or project/, else by name in <run_dir>/<sub>/."""
        p = Path(ref)
        cands = [p] if p.is_absolute() else [self.run_dir / p, self.run_dir / "project" / p]
        cands.append(self.run_dir / sub / p.name)
        return next((c for c in cands if c.is_file()), None)

    def script(self, ref: str) -> Path | None:
        return self.path(ref, "scripts")

    def text_label(self, text: str, name: str = "") -> int:
        idx, pipe = bool(INDEX_RE.search(text)), bool(PIPELINE_RE.search(text))
        if idx and pipe:   # one script does both: the name decides only when it says which
            return INDEX if INDEX_NAME.search(name) and not PIPELINE_NAME.search(name) else MIXED
        if idx:
            return INDEX
        if pipe:
            return PIPELINE
        if SETUP_RE.search(text) or SETUP_NAME.search(name):
            return SETUP
        if DOWNLOAD_RE.search(text) or DOWNLOAD_NAME.search(name):
            return DOWNLOAD
        if INDEX_NAME.search(name) and not PIPELINE_NAME.search(name):
            return INDEX
        if PIPELINE_NAME.search(name):
            return PIPELINE
        return OTHER

    def file_label(self, path: Path, depth: int = 0) -> int:
        key = (str(path), depth)
        if key not in self.cache:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            label = self.text_label(text, path.name)
            if path.suffix == ".sh" and depth < 2:   # a wrapper: what it runs decides
                labels = {label} | {self.file_label(s, depth + 1) for r in SCRIPT_REF.findall(self.expand(text)) if (s := self.script(r))}
                # an orchestrator that builds indexes and runs pipelines counts as build + run
                label = MIXED if INDEX in labels and labels & {PIPELINE, MIXED} else max(labels)
            self.cache[key] = label
        return self.cache[key]

    def command(self, cmd: str) -> int:
        """Label of a shell command or process command line."""
        if cmd not in self.cache:
            self.cache[cmd] = self._command(cmd)
        return self.cache[cmd]

    def _command(self, cmd: str) -> int:
        runs = self.expand(HEREDOC.sub("", cmd))   # heredoc bodies only write files; what the command runs is outside them
        labels = {OTHER}
        for ref in SCRIPT_REF.findall(runs):
            r = re.escape(ref)
            if re.search(rf"(python[\d.]*|bash|sh)\s+(-\w+\s+)*{r}|(^|[;&|(\"']\s*){r}", runs, re.M) and (p := self.script(ref)):
                labels.add(self.file_label(p))
        if re.search(r"python[\d.]*\s+(-c\s|-\s*<<)", runs) or re.search(r"python[\d.]*\s+-c\s", cmd):   # inline code
            labels.add(self.text_label(cmd))
        labels |= {label for rx, label in ((SETUP_RE, SETUP), (DOWNLOAD_RE, DOWNLOAD)) if rx.search(runs)}
        return max(labels)


class Timeline:
    def __init__(self, start: float, end: float):
        self.start, self.n = start, max(int(end - start) + 1, 1)
        self.label = bytearray(self.n)
        self.llm = bytearray(self.n)

    def mark(self, a: float, b: float, label: int) -> None:
        i, j = max(int(a - self.start), 0), min(int(b - self.start) + 1, self.n)
        for k in range(i, j):
            if label == LLM:
                self.llm[k] = 1
            if label > self.label[k]:
                self.label[k] = label


def transcript_events(rows: list[dict]):
    """(time, kind, payload) for every user/assistant row with a timestamp, in time order."""
    out = []
    for r in rows:
        if r.get("type") not in ("user", "assistant") or "timestamp" not in r:
            continue
        out.append((ts(r["timestamp"]), r))
    return sorted(out, key=lambda x: x[0])


def split(run_dir: Path, end: float | None = None) -> dict | None:
    """Seconds per category from the condition's start to `end` (default: its last transcript row)."""
    st = matrix.read_json(run_dir / "agent_state.json", {}) or {}
    rows = read_rows(run_dir)
    events = transcript_events(rows)
    if not st.get("started") or not events:
        return None
    start = ts(st["started"])
    end = end or events[-1][0]
    tl, cls = Timeline(start, end), Classifier(run_dir)

    calls: dict[str, tuple[float, dict]] = {}
    bg: list[tuple[float, str, int]] = []   # `nohup ... > log &` jobs: launch time, log path, label
    cli_bg: dict[str, tuple[float, int, str | None]] = {}   # the CLI's background tasks: id -> start, label, log
    done_at = {m.group(1): t for t, r in events if isinstance(r["message"].get("content"), str)
               for m in re.finditer(r"<task-id>(\w+)</task-id>(?:(?!</task-notification>).)*<status>(?:completed|failed|killed)",
                                    r["message"]["content"], re.S)}
    prev = start
    for t, r in events:
        content = r.get("message", {}).get("content")
        if r["type"] == "assistant":
            tl.mark(prev, t, LLM)
            for c in content if isinstance(content, list) else []:
                if c.get("type") == "tool_use":
                    calls[c["id"]] = (t, c)
                    cmd = HEREDOC.sub("", str(c["input"].get("command") or ""))
                    for m in BG_LAUNCH.finditer(cmd):
                        bg.append((t, cls.expand(m.group(2)).strip("'\""), cls.command(m.group(1))))
        else:
            for c in content if isinstance(content, list) else []:
                if c.get("type") == "tool_result" and c.get("tool_use_id") in calls:
                    t0, use = calls.pop(c["tool_use_id"])
                    cmd = str(use["input"].get("command") or "")
                    label = cls.command(cmd) if use["name"] == "Bash" else OTHER
                    tl.mark(t0, t, label)
                    task = (r.get("toolUseResult") or {}).get("backgroundTaskId") if isinstance(r.get("toolUseResult"), dict) else None
                    if task:   # the CLI's own background mode: runs until its task notification
                        log = re.search(r"(?:tee(?:\s+-a)?|>>?)\s*(\S+\.(?:log|out|txt))", HEREDOC.sub("", cmd))
                        cli_bg[task] = (t0, label, cls.expand(log.group(1)) if log else None)
        prev = t
    # the runner records each resume: from the last transcript row before it to the resume, the session was dead
    for r in st.get("resumed") or []:
        at = ts(r["at"])
        last = max((t for t, _ in events if t < at), default=None)
        if last is not None:
            tl.mark(last, at, DOWN)
    for w in st.get("limit_waits") or []:   # the runner slept until a usage limit reset
        tl.mark(ts(w["from"]), ts(w["until"]), LIMIT)

    activity = run_dir / "activity.jsonl"
    source = "measured" if activity.exists() else "reconstructed"
    if activity.exists():
        samples = [json.loads(l) for l in activity.read_text(encoding="utf-8").splitlines() if l.strip()]
        for s, nxt in zip(samples, samples[1:] + [None]):
            a = ts(s["time"])
            b = min(ts(nxt["time"]), a + 15) if nxt else a + 5
            for p in s["procs"]:
                tl.mark(a, b, cls.command(p["cmd"]))
    else:
        for task, (t, label, log) in cli_bg.items():
            stop = done_at.get(task)
            if stop is None and log and (path := cls.path(log, "logs")):
                stop = path.stat().st_mtime
            if stop and stop > t:
                tl.mark(t, stop, label)
        for i, (t, log, label) in enumerate(bg):
            path = cls.path(log, "logs")
            if path is None:
                continue
            stop = path.stat().st_mtime
            later = [t2 for t2, log2, *_ in bg[i + 1:] if log2 == log]
            if later:
                stop = min(stop, later[0])
            if stop > t:
                tl.mark(t, stop, label)

    secs = {k: 0 for k in CATEGORIES}
    down = limit = 0
    for label in tl.label:
        if label == DOWN:
            down += 1
        elif label == LIMIT:
            limit += 1
        else:
            secs[NAME[label]] += 1
    llm_total = sum(tl.llm)
    per_message = {r["message"]["id"]: (ts(r["timestamp"]), r["message"].get("usage") or {}) for r in rows
                   if r.get("type") == "assistant" and r.get("message", {}).get("id") and "timestamp" in r}
    usage = [u for t, u in per_message.values() if t <= end]
    tokens = {k: sum(u.get(f, 0) for u in usage) for k, f in
              (("input", "input_tokens"), ("cache_write", "cache_creation_input_tokens"),
               ("cache_read", "cache_read_input_tokens"), ("output", "output_tokens"))}
    return {"source": source, "window_s": tl.n - down - limit, "down_s": down, "limit_wait_s": limit, **{f"{k}_s": v for k, v in secs.items()},
            "llm_total_s": llm_total, "tokens": tokens}


def first_bm25_end(run_dir: Path) -> float | None:
    first = next((e["time"] for e in matrix.status_log(run_dir)
                  if e["pipeline"] == "bm25" and e["key"] == "dl19" and e["state"] == "done"), None)
    return ts(first) if first else None


def condition_split(run_dir: Path) -> dict:
    end = first_bm25_end(run_dir)
    return {"to_first_bm25": split(run_dir, end) if end else None, "whole_run": split(run_dir)}


def markdown(splits: dict[str, dict], fmt_s) -> list[str]:
    def cell(s: int, total: int) -> str:
        return f"{fmt_s(s)} ({100 * s / total:.0f}%)" if total else "–"

    lines = []
    for key, title in (("to_first_bm25", "start → first BM25 run on DL19"), ("whole_run", "whole run")):
        lines += [f"## Time split: {title}", "",
                  "| condition | source | window | agentic coding | index building | pipeline running | build + run | env setup | "
                  "downloads | other | LLM incl. overlap | input + cache tokens | output tokens |", "|" + "---|" * 13]
        for cond, sp in splits.items():
            s = sp.get(key)
            if not s:
                lines.append(f"| {cond} |" + " – |" * 12)
                continue
            w, t = s["window_s"], s["tokens"]
            lines.append(f"| {cond} | {s['source']} | {fmt_s(w)} | " + " | ".join(cell(s[f'{c}_s'], w) for c in CATEGORIES) +
                         f" | {fmt_s(s['llm_total_s'])} | {t['input'] + t['cache_write'] + t['cache_read']:,} | {t['output']:,} |")
        lines.append("")
    return lines + [
        "How to read the time split:",
        "- Each second gets one label: index building > build + run > pipeline running > downloads > env setup > agentic coding "
        "(model generating) > other. Model turns taken while a job runs count as that job; LLM incl. overlap is all model time.",
        "- other = polling, inspecting files, harness and API waits. Time between a killed session and its resume, and waits "
        "for a subscription usage limit to reset (when no job ran meanwhile), are left out.",
        "- measured = sampled processes (activity.jsonl); reconstructed = transcript plus background-job logs (approximate).",
        "- build + run = one script (or wrapper) that both builds indexes and runs pipelines; there is no reliable way to split it.", ""]


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, json.dumps(condition_split(Path(arg)), indent=2))
