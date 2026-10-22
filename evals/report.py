#!/usr/bin/env python3
"""Summarise one eval iteration: time, human/LLM interaction, tokens, compute, reuse and rigour.

Writes report.md and report.json into the iteration folder. run_evals.py calls it at the end of
every batch; run it by hand for an older or unfinished iteration.

Per eval it reports
    time        wall time, of which LLM time (model generating, from the transcripts) and the rest (tools and code)
    interaction human messages, LLM turns, tool calls by tool
    tokens      input, cache write, cache read, output; the CLI's cost estimate (costBasis may be
                "unknown" for non-Anthropic gateways, so treat it as a token-volume indicator)
    compute     CPU seconds and peak RSS of the agent and every process it ran (from wait4 in
                run_evals.py; missing for runs started before that was added), GPU visibility
    reuse       prebuilt artifacts / caches loaded vs indexes built, from the code the agent wrote
    source      skill reads vs library-source reads (see source_reads.py)
    rigour      pt.Experiment calls with a baseline, a significance test and a correction
    same result outcome numbers shared with the previous iteration of the same eval
    progress    for evals still running: started, elapsed, idle, turns, and an ETA from the same eval's
                last finished run, else this batch's finished evals, else the turn rate

Examples
    python evals/report.py                                   # latest iteration
    python evals/report.py ../pyterrier-skills-evals/iteration-4 --deadline 12:42
    python evals/report.py --watch 60                        # live status every minute
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_reads import DEFAULT_WORKSPACE, classify  # noqa: E402

PREBUILT = re.compile(r"\.from_hf\(|Artifact\.load\(|\.example\(\)|from_dataset\(|from_url\(")
# loads that are not from Hugging Face (hf: ids, huggingface.co / hf.co URLs and from_hf are)
NON_HF_PREBUILT = re.compile(r"\.example\(\)|from_dataset\(|(from_url|Artifact\.load)\(\s*[\"']\w+://(?!(www\.)?(huggingface|hf)\.co)")
CACHES = re.compile(r"\b(ScorerCache|RetrieverCache|IndexerCache|KeyValueCache|Lazy)\(")
BUILDS = re.compile(r"IterDictIndexer\(|\.indexer\(|>>\s*\w*index\w*\)?\s*\)?\.index\(|\)\.index\(|\.index\(\s*\w+\.get_corpus_iter")
EXPERIMENT = re.compile(r"pt\.Experiment\(")
NUMBER = re.compile(r"\b0\.\d{3,4}\b")
STALLED_AFTER_S = 15 * 60
# user-role messages the CLI writes itself: recovery after a cut-off reply, and compaction summaries
CLI_GENERATED = ("Your response above was cut off", "This session is being continued from a previous")


def transcripts(run: Path) -> list[Path]:
    return sorted(run.glob("claude-config/projects/**/*.jsonl"))


def read_rows(run: Path) -> list[dict]:
    rows = []
    for t in transcripts(run):
        rows += [json.loads(line) for line in t.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def llm_seconds(run: Path) -> float | None:
    """Time the model spent generating, from the transcripts: each assistant row closes the span that began at
    the row before it (a prompt or tool result), and spans are merged across files (subagents, resumes).
    result.json's duration_api_ms covers only the last segment of a session that continued after a background
    task notification (e.g. 0 min for a 2.6 h run), so it is not used."""
    spans = []
    for path in transcripts(run):
        prev = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") not in ("user", "assistant") or not r.get("timestamp"):
                continue
            t = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp()
            if r["type"] == "assistant" and prev is not None and t > prev:
                spans.append((prev, t))
            prev = t
    total, end = 0.0, None
    for a, b in sorted(spans):
        if end is None or a > end:
            total, end = total + b - a, b
        elif b > end:
            total, end = total + b - end, b
    return total if spans else None


def agent_code(rows: list[dict]) -> list[str]:
    """Code the agent wrote or ran: Write contents, Edit replacements, Bash commands."""
    out = []
    for r in rows:
        content = r.get("message", {}).get("content")
        if r.get("type") != "assistant" or not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use":
                i = c["input"]
                out += [str(i.get(k)) for k in ("content", "new_string", "command") if i.get(k)]
    return out


def pending_command(rows: list[dict]) -> str | None:
    """The last tool call if it has no result yet (a long script or install still running)."""
    calls, answered = {}, set()
    for r in rows:
        content = r.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use":
                calls[c["id"]] = str(c["input"].get("command") or c["input"].get("file_path") or c["name"])
            elif c.get("type") == "tool_result":
                answered.add(c.get("tool_use_id"))
    open_calls = [cid for cid in calls if cid not in answered]
    return calls[open_calls[-1]][:80].replace("\n", " ") if open_calls else None


def experiment_flags(code: list[str]) -> dict:
    """Look at every pt.Experiment(...) call (first ~600 chars after it) for baseline/test/correction."""
    calls = []
    for text in code:
        for m in EXPERIMENT.finditer(text):
            calls.append(text[m.start():m.start() + 600])
    corr = Counter(m.group(1) for c in calls for m in re.finditer(r"correction\s*=\s*['\"]([\w-]+)", c))
    return {
        "calls": len(calls),
        "with_baseline": sum("baseline" in c for c in calls),
        "with_test": sum(bool(re.search(r"\btest\s*=", c)) or "baseline" in c for c in calls),
        "corrections": dict(corr),
    }


def outcome_numbers(run: Path) -> set[str]:
    findings = run / "findings.md"
    if not findings.exists():
        return set()
    text = findings.read_text(encoding="utf-8")
    m = re.search(r"#+\s*1\..*?(?=\n#+\s*2\.)", text, re.S)
    return set(NUMBER.findall(m.group(0) if m else ""))


def eval_report(run: Path, previous: Path | None) -> dict:
    timing = json.loads((run / "timing.json").read_text()) if (run / "timing.json").exists() else {}
    result = {}
    if (run / "result.json").exists():
        try:
            result = json.loads((run / "result.json").read_text())
        except json.JSONDecodeError:
            pass
    rows = read_rows(run)

    texts = [r["message"]["content"] for r in rows if r.get("type") == "user" and isinstance(r.get("message", {}).get("content"), str)]
    auto = sum(t.startswith(CLI_GENERATED) for t in texts)   # the CLI's own continuations, not a person
    human = len(texts) - auto
    llm_turns = len({r.get("message", {}).get("id") or r.get("uuid") for r in rows if r.get("type") == "assistant"})
    tools, skill_reads, source_reads = Counter(), 0, 0
    for r in rows:
        content = r.get("message", {}).get("content")
        if r.get("type") != "assistant" or not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use":
                tools[c["name"]] += 1
                i = c["input"]
                kind = classify(c["name"], str(i.get("command") or i.get("file_path") or i.get("path") or i.get("pattern") or ""))
                skill_reads += kind is not None and kind[0] == "skill"
                source_reads += kind is not None and kind[0] == "source"

    # one usage per API message (a message is split over several transcript rows); summed over every
    # session, so resumed and retried runs count fully. result.json only covers the last session.
    per_message = {r["message"]["id"]: r["message"].get("usage") or {} for r in rows
                   if r.get("type") == "assistant" and r.get("message", {}).get("id")}
    usage = ({k: sum(u.get(k, 0) for u in per_message.values()) for k in
              ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
             if per_message else result.get("usage") or {})
    tokens = {
        "input": usage.get("input_tokens", 0),
        "cache_write": usage.get("cache_creation_input_tokens", 0),
        "cache_read": usage.get("cache_read_input_tokens", 0),
        "output": usage.get("output_tokens", 0),
    }
    tokens["total"] = sum(tokens.values())
    cost_basis = {m.get("costBasis") for m in (result.get("modelUsage") or {}).values()}

    stamps = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in rows if r.get("timestamp")]
    now = datetime.now(timezone.utc)
    started = min(stamps) if stamps else (
        datetime.fromtimestamp((run / "prompt.md").stat().st_mtime, timezone.utc) if (run / "prompt.md").exists() else None)
    last = max(stamps) if stamps else started
    finished = bool(timing)
    if finished:
        state = "done" if (run / "findings.md").exists() else "no findings"
    elif (run / "findings.md").exists():
        state = "writing findings"
    elif pending_command(rows):
        state = "running a command"   # the transcript is silent until the tool returns
    elif last and (now - last).total_seconds() > STALLED_AFTER_S:
        state = "waiting on LLM / stalled?"
    else:
        state = "running"
    progress = {
        "state": state,
        "started": started.isoformat() if started else None,
        "last_activity": last.isoformat() if last else None,
        "elapsed_s": round((now - started).total_seconds()) if started and not finished else None,
        "idle_s": round((now - last).total_seconds()) if last and not finished else None,
        "pending_command": pending_command(rows) if not finished else None,
        "findings_written": datetime.fromtimestamp((run / "findings.md").stat().st_mtime, timezone.utc).isoformat()
        if (run / "findings.md").exists() else None,
    }

    wall = timing.get("total_duration_seconds") or (result.get("duration_ms", 0) / 1000) or None
    api = (llm_seconds(run) or result.get("duration_api_ms", 0) / 1000) if wall else None
    code = agent_code(rows)
    prev_numbers = outcome_numbers(previous) if previous else set()
    numbers = outcome_numbers(run)
    return {
        "eval": run.name,
        "status": "ok" if finished and (run / "findings.md").exists() else state,
        "progress": progress,
        "time": {"wall_s": round(wall) if wall else None, "llm_s": round(api) if wall else None,
                 "tools_and_code_s": round(max(wall - api, 0)) if wall else None},
        "interaction": {"human_messages": human, "cli_continuations": auto, "llm_turns": llm_turns, "num_turns": result.get("num_turns"),
                        "tool_calls": sum(tools.values()), "by_tool": dict(tools)},
        "tokens": tokens,
        "cost_estimate_usd": result.get("total_cost_usd"),
        "cost_basis": sorted(b for b in cost_basis if b) or None,
        "compute": {k: timing.get(k) for k in ("cpu_user_s", "cpu_system_s", "max_rss_mb", "gpu")},
        "reuse": {
            "prebuilt_loads": sum(len(PREBUILT.findall(t)) for t in code),
            "non_hf_prebuilt_loads": sum(len(NON_HF_PREBUILT.findall(t)) for t in code),
            "cache_uses": sum(len(CACHES.findall(t)) for t in code),
            "index_builds": sum(len(BUILDS.findall(t)) for t in code),
        },
        "source": {"skill_reads": skill_reads, "source_reads": source_reads},
        "rigour": experiment_flags(code),
        "same_result": {
            "previous": previous.parent.name if previous else None,
            "shared_numbers": len(numbers & prev_numbers),
            "numbers": len(numbers),
            "previous_numbers": len(prev_numbers),
        },
    }


def eval_max_turns(run: Path, meta: dict) -> int:
    """The eval's own turn limit (recorded by run_evals.py in eval_metadata.json), else the batch's."""
    try:
        return json.loads((run / "eval_metadata.json").read_text(encoding="utf-8"))["max_turns"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return meta.get("max_turns", 150)


def previous_run(iteration: Path, run: Path) -> Path | None:
    n = int(iteration.name.split("-")[1])
    for k in range(n - 1, 0, -1):
        cand = iteration.parent / f"iteration-{k}" / run.name
        if (cand / "findings.md").exists():
            return cand
    return None


def fmt_s(s) -> str:
    return "–" if s is None else f"{s // 60}m{s % 60:02d}s"


def clock(iso: str | None) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%H:%M") if iso else "–"


def finished_wall(run: Path | None) -> float | None:
    timing = run / "timing.json" if run else None
    if timing and timing.exists():
        return json.loads(timing.read_text()).get("total_duration_seconds")
    return None


def add_eta(iteration: Path, evals: list[dict], meta: dict, deadline: datetime | None) -> None:
    """Fill progress.eta / eta_basis / worst_case for evals that have not finished."""
    done = [e for e in evals if e["time"]["wall_s"]]
    batch_wall = statistics.median(e["time"]["wall_s"] for e in done) if done else None
    batch_turns = statistics.median(e["interaction"]["llm_turns"] for e in done) if done else None
    # how long finished agents kept going after writing findings.md (final message, last checks)
    tails = [(datetime.fromisoformat(e["progress"]["started"]) + timedelta(seconds=e["time"]["wall_s"])
              - datetime.fromisoformat(e["progress"]["findings_written"])).total_seconds()
             for e in done if e["progress"].get("findings_written") and e["progress"].get("started")]
    tail = max(statistics.median(tails), 60) if tails else 180
    for e in evals:
        p = e["progress"]
        if e["time"]["wall_s"] or not p["started"]:
            continue
        started = datetime.fromisoformat(p["started"])
        elapsed, turns = p["elapsed_s"], e["interaction"]["llm_turns"]
        pace = elapsed / turns if turns else None
        prev_wall = finished_wall(previous_run(iteration, iteration / e["eval"]))
        max_turns = eval_max_turns(iteration / e["eval"], meta)
        expected, basis = (prev_wall, "previous iteration") if prev_wall else (batch_wall, "median of finished evals")
        if p["state"].endswith("stalled?"):
            eta, basis = None, "no transcript activity for 15+ min: a slow LLM reply, or the agent was killed"
        elif p.get("findings_written"):
            eta = datetime.fromisoformat(p["findings_written"]) + timedelta(seconds=tail)
            basis = f"findings written; finished evals ended ~{round(tail / 60)} min later"
        elif expected and elapsed < expected:
            eta = started + timedelta(seconds=expected)
        elif batch_turns and turns and turns < batch_turns:
            eta, basis = started + timedelta(seconds=elapsed * batch_turns / turns), "turn rate"
        else:
            eta, basis = None, "longer than any finished eval; findings not written yet"
        worst = (started + timedelta(seconds=pace * max_turns)
                 if pace and not p["state"].endswith("stalled?") else None)
        p.update(eta=eta.isoformat() if eta else None, eta_basis=basis, pace_s_per_turn=round(pace) if pace else None,
                 worst_case=worst.isoformat() if worst else None, max_turns=max_turns,
                 after_deadline=bool(deadline and (eta or worst) and (eta or worst) > deadline))


def running_table(evals: list[dict], deadline: datetime | None) -> list[str]:
    running = [e for e in evals if not e["time"]["wall_s"]]
    if not running:
        return []
    lines = ["## Running", ""]
    if deadline:
        lines.append(f"Deadline: {deadline.astimezone():%H:%M}. Evals still running then are cut off.")
        lines.append("")
    lines += ["| eval | state | started | elapsed | idle | turns | pace | tool calls | ETA | basis | worst case |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in running:
        p = e["progress"]
        flag = (" ⚠ after deadline" if p.get("eta") else " ⚠ may run past deadline") if p.get("after_deadline") else ""
        lines.append(
            f"| {e['eval']} | {p['state']} | {clock(p['started'])} | {fmt_s(p['elapsed_s'])} | {fmt_s(p['idle_s'])} | "
            f"{e['interaction']['llm_turns']}/{p.get('max_turns', 150)} | "
            f"{str(p['pace_s_per_turn']) + ' s/turn' if p.get('pace_s_per_turn') else '–'} | {e['interaction']['tool_calls']} | "
            f"{clock(p.get('eta'))}{flag} | {p.get('eta_basis', '–')} | {clock(p.get('worst_case'))} |")
    return lines + [""]


def markdown(iteration: Path, evals: list[dict], meta: dict, deadline: datetime | None = None) -> str:
    t = lambda key: sum(e["tokens"][key] for e in evals)  # noqa: E731
    cpu = [e["compute"]["cpu_user_s"] + e["compute"]["cpu_system_s"] for e in evals
           if e["compute"].get("cpu_user_s") is not None]
    lines = [
        f"# Eval report: {iteration.name}",
        "",
        f"- Model: {meta.get('model') or 'default'} · effort: {meta.get('effort') or 'default'} · jobs: {meta.get('jobs', '–')}",
        f"- Batch wall time: {fmt_s(meta.get('wall_s'))} · evals: {len(evals)} "
        f"({sum(e['status'] == 'ok' for e in evals)} with findings)",
        f"- Human messages: {sum(e['interaction']['human_messages'] for e in evals)} · "
        f"LLM turns: {sum(e['interaction']['llm_turns'] for e in evals)} · "
        f"tool calls: {sum(e['interaction']['tool_calls'] for e in evals)}",
        f"- Tokens: {t('total'):,} total (input {t('input'):,}, cache write {t('cache_write'):,}, "
        f"cache read {t('cache_read'):,}, output {t('output'):,})",
        f"- CLI cost estimate: ${sum(e['cost_estimate_usd'] or 0 for e in evals):.2f} "
        f"(cost basis: {', '.join(sorted({b for e in evals for b in (e['cost_basis'] or [])})) or 'n/a'}; "
        "not a bill for non-Anthropic gateways)",
        f"- CPU time: {fmt_s(round(sum(cpu))) if cpu else 'not recorded (run started before wait4 accounting)'}"
        f" · GPU: {meta.get('gpu', 'not recorded')}",
        "",
        *running_table(evals, deadline),
        "## All evals",
        "",
        "| eval | status | wall | LLM | tools+code | human | turns | tool calls | tokens | CPU | peak RSS | "
        "prebuilt / cache / builds | skill / source reads | Experiment: baseline · correction | same as prev |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in evals:
        c, r, g, s = e["compute"], e["reuse"], e["rigour"], e["same_result"]
        cpu_s = (c["cpu_user_s"] or 0) + (c["cpu_system_s"] or 0) if c.get("cpu_user_s") is not None else None
        rig = (f"{g['with_baseline']}/{g['calls']} · {', '.join(f'{k}×{v}' for k, v in g['corrections'].items()) or 'none'}"
               if g["calls"] else "no pt.Experiment")
        same = f"{s['shared_numbers']}/{s['numbers']} vs {s['previous']}" if s["previous"] else "–"
        lines.append(
            f"| {e['eval']} | {e['status']} | {fmt_s(e['time']['wall_s'])} | {fmt_s(e['time']['llm_s'])} | "
            f"{fmt_s(e['time']['tools_and_code_s'])} | {e['interaction']['human_messages']} | "
            f"{e['interaction']['llm_turns']} | {e['interaction']['tool_calls']} | {e['tokens']['total']:,} | "
            f"{fmt_s(round(cpu_s)) if cpu_s is not None else '–'} | {c.get('max_rss_mb') or '–'} MB | "
            f"{r['prebuilt_loads']} / {r['cache_uses']} / {r['index_builds']} | "
            f"{e['source']['skill_reads']} / {e['source']['source_reads']} | {rig} | {same} |"
        )
    lines += [
        "",
        "How to read it:",
        "- **LLM** is the time the model spent generating, measured from the transcripts; **tools+code** is wall time minus that: running scripts, installs, downloads.",
        "- **prebuilt / cache / builds** counts calls in the code the agent wrote (from_hf, Artifact.load, "
        "*Cache, Lazy vs indexer/.index). More prebuilt and cache, fewer builds, is better.",
        "- **source reads** count site-packages reads and inspect/help probes: each is something the skill did not say.",
        "- **Experiment** shows how many pt.Experiment calls had a baseline, and which corrections were used.",
        "- **ETA** uses the same eval's last finished run, else the median of evals finished in this batch, "
        "else the time after findings.md is written, else extrapolates from turns; **worst case** is when it "
        "would hit max-turns at the current pace (⚠ when the ETA, or failing that the worst case, is after the deadline). "
        "**running a command** means a tool call has not returned yet; **waiting on LLM / stalled?** means no "
        "transcript activity for 15 minutes and no command pending.",
        "- **same as prev** counts the outcome numbers (0.xxx) in findings §1 that also appear in the previous "
        "iteration's findings for the same eval: a rough reproducibility check, not a proof.",
    ]
    return "\n".join(lines) + "\n"


def build(iteration: Path, deadline: datetime | None = None) -> tuple[dict, list[dict]]:
    meta = {}
    if (iteration / "run_meta.json").exists():
        meta = json.loads((iteration / "run_meta.json").read_text())
    runs = sorted(iteration.glob("eval-*"), key=lambda p: int(p.name.split("-")[1]))
    evals = [eval_report(run, previous_run(iteration, run)) for run in runs]
    add_eta(iteration, evals, meta, deadline)
    return meta, evals


def write_report(iteration: Path, deadline: datetime | None = None) -> Path:
    meta, evals = build(iteration, deadline)
    (iteration / "report.json").write_text(json.dumps({"meta": meta, "evals": evals}, indent=2), encoding="utf-8")
    (iteration / "report.md").write_text(markdown(iteration, evals, meta, deadline), encoding="utf-8")
    return iteration / "report.md"


def parse_deadline(text: str | None) -> datetime | None:
    """HH:MM today in local time (tomorrow if that time has already passed by more than 12 hours)."""
    if not text:
        return None
    h, m = map(int, text.split(":"))
    now = datetime.now().astimezone()
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return d + timedelta(days=1) if (now - d).total_seconds() > 12 * 3600 else d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iteration", type=Path, nargs="?", help="iteration-N folder (default: the latest)")
    ap.add_argument("--deadline", help="HH:MM when the batch will be cut off (e.g. a launcher timeout)")
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="reprint the running table every N seconds")
    args = ap.parse_args()
    iteration = args.iteration or max(DEFAULT_WORKSPACE.glob("iteration-*"), key=lambda p: int(p.name.split("-")[1]))
    deadline = parse_deadline(args.deadline)
    if not args.watch:
        print(write_report(iteration, deadline).read_text(encoding="utf-8"))
        return 0
    try:
        while True:
            write_report(iteration, deadline)
            _, evals = build(iteration, deadline)
            done = sum(bool(e["time"]["wall_s"]) for e in evals)
            print(f"\n{datetime.now():%H:%M:%S}  {iteration.name}: {done}/{len(evals)} finished")
            print("\n".join(running_table(evals, deadline)) or "all finished")
            if done == len(evals):
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
