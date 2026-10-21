#!/usr/bin/env python3
"""Live status of a pipeline-matrix batch: pipelines × datasets, with an ETA per condition.

A cell `x/N ▶k ✗f` means x of the N conditions have a valid run file for it, k are building or
running it now (from the agent's status.json), f marked it failed. A NanoBEIR cell is done for a
condition only when all 13 sub-dataset runs exist. `--condition <id>` shows one condition's cells
(✓ done, ▶ active, ✗ failed, · pending; NanoBEIR as k/13).

ETA per condition: each cell has a cost weight (pipelines.json: pipeline weight × dataset weight, plus
the MS MARCO index-build weight). ETA = start + elapsed × total weight ÷ weight done. Once a condition
of this batch (or else the previous batch) has finished, its measured seconds per weight unit are used
for the remaining weight instead. Worst case = when the agent would hit max-turns at its current pace,
counting the nudges it has left.

Turns `531/1225 · nudge 2/3`: LLM turns so far / the most it can reach. --max-turns is a limit per
attempt, not per condition: the first prompt is attempt 0, and each nudge starts another attempt with a
fresh max-turns. A nudge is the runner's automatic follow-up when the agent stops (or hits max-turns)
with cells still missing: it resumes the same session with a message listing those cells, at most
--max-nudges times, and counts as a human message. The cap is the turns the finished attempts actually
used plus max-turns for the running attempt and each nudge left. Gateway-error retries also start an
attempt but are not nudges (shown as `+k retries`). "stalled?" = no transcript activity for 15 min and no command pending.

Examples
    python evals/pipeline_status.py                      # latest pipelines-N
    python evals/pipeline_status.py --watch 60 --deadline 23:30
    python evals/pipeline_status.py --condition skills
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipelines"))
import matrix  # noqa: E402
from report import fmt_s, parse_deadline, pending_command, read_rows, STALLED_AFTER_S  # noqa: E402

CACHE_FILE = ".run_check_cache.json"


def ts(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def clock(d: datetime | None) -> str:
    return d.astimezone().strftime("%d %H:%M") if d else "–"


def sec_per_weight(run_dirs: list[Path], spec: dict) -> float | None:
    """Seconds per unit of cell weight, measured on finished conditions."""
    rates = []
    for rd in run_dirs:
        st = matrix.read_json(rd / "agent_state.json", {}) or {}
        timing = matrix.read_json(rd / "timing.json", {}) or {}
        if st.get("state") != "finished" or not timing.get("total_duration_seconds"):
            continue
        cells = matrix.condition_cells(rd, spec)
        done_keys = {(e["pipeline"], e["key"]) for e in matrix.status_log(rd) if e["state"] == "done"}
        w = sum(c["weight"] for c in cells if (c["pipeline"], c["key"]) in done_keys)
        if w:
            rates.append(timing["total_duration_seconds"] / w)
    return statistics.median(rates) if rates else None


def condition_row(rd: Path, spec: dict, cache: dict, rate: float | None, meta: dict) -> dict:
    cells = matrix.condition_cells(rd, spec)
    states = matrix.cell_states(rd, cells, cache, matrix.expected_queries(spec))
    st = matrix.read_json(rd / "agent_state.json", {}) or {}
    now = datetime.now(timezone.utc)
    total_w = sum(c["weight"] for c in cells)
    done_w = sum(c["weight"] for c in cells if states[(c["pipeline"], c["key"])] in ("done", "failed"))
    rows = read_rows(rd)
    stamps = [ts(r["timestamp"].replace("Z", "+00:00")) for r in rows if r.get("timestamp")]
    started = ts(st.get("started")) or (min(stamps) if stamps else None)
    last = max(stamps) if stamps else started
    turns = len({r.get("message", {}).get("id") or r.get("uuid") for r in rows if r.get("type") == "assistant"})
    pending = pending_command(rows)
    row = {"condition": rd.name, "state": st.get("state", "queued"), "started": started, "turns": turns,
           "done": sum(s == "done" for s in states.values()), "failed": sum(s == "failed" for s in states.values()),
           "cells": len(cells), "states": states, "cells_list": cells, "eta": None, "basis": "–", "worst": None,
           "elapsed": None, "idle": None, "finished": ts(st.get("finished")), "outcome": st.get("outcome")}
    row["turn_cap"], row["nudge"], row["retries"] = turn_budget(rd, st, meta)
    if row["state"] != "running" or not started:
        return row
    elapsed = (now - started).total_seconds()
    idle = (now - last).total_seconds() if last else None
    row.update(elapsed=round(elapsed), idle=round(idle) if idle is not None else None)
    if pending:
        row["state"] = "running a command"
    elif idle and idle > STALLED_AFTER_S:
        row["state"] = "stalled?"
    if rate:
        row["eta"], row["basis"] = now + timedelta(seconds=rate * (total_w - done_w)), "measured s/weight"
    elif done_w:
        row["eta"], row["basis"] = started + timedelta(seconds=elapsed * total_w / done_w), "cell weights"
    else:
        row["basis"] = "no cell finished yet"
    if turns:
        row["worst"] = now + timedelta(seconds=elapsed / turns * max(row["turn_cap"] - turns, 0))
    return row


def turn_budget(rd: Path, st: dict, meta: dict) -> tuple[int, int, int]:
    """(turn cap, nudge in progress or used, API retries). Each attempt (prompt, nudge or retry) may use
    max_turns; finished attempts count with the turns they actually used (result-<n>.json)."""
    max_turns, max_nudges = meta.get("max_turns", 400), meta.get("max_nudges", 3)
    attempts, retries = st.get("attempts") or 0, st.get("api_retries") or 0
    used = sum((matrix.read_json(rd / f"result-{i}.json", {}) or {}).get("num_turns") or 0 for i in range(attempts))
    if st.get("state") == "finished":
        return used, st.get("nudges") or 0, retries
    nudge = max(attempts - retries, 0)   # agent_state is written before the runner counts the nudge it is starting
    return used + max_turns * (1 + max(max_nudges - nudge, 0)), nudge, retries


def turns_cell(r: dict, meta: dict) -> str:
    if r["state"] == "queued":
        return "–"
    retries = f" +{r['retries']} retries" if r["retries"] else ""
    if r["state"] == "finished":
        return f"{r['turns']} · {r['nudge']} nudges{retries}"
    return f"{r['turns']}/{r['turn_cap']} · nudge {r['nudge']}/{meta.get('max_nudges', 3)}{retries}"


def matrix_table(spec: dict, rows: list[dict]) -> list[str]:
    cols = [d["id"] for d in spec["datasets"]]
    lines = ["| pipeline | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for p in spec["pipelines"]:
        out = []
        for col in cols:
            n = done = active = failed = 0
            for r in rows:
                ss = [s for (pp, k), s in r["states"].items() if pp == p["id"] and _col(r, pp, k) == col]
                if not ss:
                    continue
                n += 1
                done += all(s == "done" for s in ss)
                active += any(s in ("building", "running") for s in ss) and r["state"] not in ("finished", "queued")
                failed += any(s == "failed" for s in ss)
            out.append("–" if not n else f"{done}/{n}" + (f" ▶{active}" if active else "") + (f" ✗{failed}" if failed else ""))
        lines.append(f"| {p['id']} | " + " | ".join(out) + " |")
    return lines


def _col(row: dict, pipeline: str, key: str) -> str:
    return next(c["column"] for c in row["cells_list"] if c["pipeline"] == pipeline and c["key"] == key)


def condition_grid(spec: dict, row: dict) -> list[str]:
    cols = [d["id"] for d in spec["datasets"]]
    mark = {"done": "✓", "building": "▶ build", "running": "▶", "failed": "✗", "pending": "·"}
    lines = [f"### {row['condition']}", "", "| pipeline | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for p in spec["pipelines"]:
        out = []
        for col in cols:
            ss = [s for (pp, k), s in row["states"].items() if pp == p["id"] and _col(row, pp, k) == col]
            if not ss:
                out.append("–")
            elif len(ss) > 1:
                act = sum(s in ("building", "running") for s in ss)
                out.append(f"{sum(s == 'done' for s in ss)}/{len(ss)}" + (f" ▶{act}" if act else "")
                           + (f" ✗{sum(s == 'failed' for s in ss)}" if "failed" in ss else ""))
            else:
                out.append(mark.get(ss[0], ss[0]))
        lines.append(f"| {p['id']} | " + " | ".join(out) + " |")
    return lines


def render(iteration: Path, spec: dict, deadline: datetime | None, only: str | None) -> tuple[str, bool]:
    meta = matrix.read_json(iteration / "run_meta.json", {}) or {}
    cache = matrix.read_json(iteration / CACHE_FILE, {}) or {}
    dirs = matrix.condition_dirs(iteration, spec)
    rate, rate_prev = sec_per_weight(dirs, spec), None
    # the previous batch of the same name (pipelines-3 for pipelines-4, sonnet-5.5-pipeline-1 has none before it)
    stem, _, n = iteration.name.rpartition("-")
    prev = iteration.parent / f"{stem}-{int(n) - 1}" if n.isdigit() else None
    if rate is None and prev and prev.exists():   # fall back to the previous batch's measured rate
        rate_prev = sec_per_weight(matrix.condition_dirs(prev, spec), spec)
    rows = [condition_row(rd, spec, cache, rate or rate_prev, meta) for rd in dirs]
    if rate is None and rate_prev:
        for r in rows:
            if r["eta"] is not None and r["basis"] == "measured s/weight":
                r["basis"] = "previous batch s/weight"
    (iteration / CACHE_FILE).write_text(json.dumps(cache), encoding="utf-8")

    running = [r["condition"] for r in rows if r["state"] not in ("finished", "queued")]
    finished = [r["condition"] for r in rows if r["state"] == "finished"]
    queued = [r for r in rows if r["state"] == "queued"]
    out = [f"{datetime.now():%Y-%m-%d %H:%M:%S}  {iteration.name}  model {meta.get('model', '?')}  "
           f"running: {', '.join(running) or 'none'}  finished: {', '.join(finished) or 'none'}  "
           f"queued: {', '.join(r['condition'] for r in queued) or 'none'}", ""]
    if only:
        out += condition_grid(spec, next(r for r in rows if r["condition"] == only)) + [""]
    else:
        out += matrix_table(spec, rows) + [""]

    out += ["| condition | state | started | elapsed | idle | turns | cells | ETA | basis | worst case |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = " ⚠" if deadline and (r["eta"] or r["worst"]) and (r["eta"] or r["worst"]) > deadline else ""
        state = f"finished ({r['outcome']})" if r["state"] == "finished" else r["state"]
        out.append(f"| {r['condition']} | {state} | {clock(r['started'])} | {fmt_s(r['elapsed'])} | {fmt_s(r['idle'])} | "
                   f"{turns_cell(r, meta)} | {r['done']}/{r['cells']}" + (f" ✗{r['failed']}" if r["failed"] else "")
                   + f" | {clock(r['eta'] or r['finished'])}{flag} | {r['basis'] if r['state'] != 'finished' else 'finished'} | {clock(r['worst'])} |")

    out += ["", f"turns = LLM turns so far / most it can reach; max-turns ({meta.get('max_turns', 400)}) applies per attempt. "
            f"nudge k/{meta.get('max_nudges', 3)} = the runner has resumed the agent k times with a list of its missing cells, "
            "because it stopped or hit max-turns before finishing."]
    durations = [(r["finished"] - r["started"]).total_seconds() for r in rows if r["finished"] and r["started"]]
    durations += [(r["eta"] - r["started"]).total_seconds() for r in rows if r["eta"] and r["started"] and not r["finished"]]
    run_etas = [r["eta"] for r in rows if r["eta"] and not r["finished"]]
    if running or queued:
        if (running and not run_etas) or (queued and not durations):
            out += ["", "Batch ETA: not enough finished cells yet."]
        else:
            base = max(run_etas) if run_etas else datetime.now(timezone.utc)
            batch = base + timedelta(seconds=len(queued) * statistics.median(durations) / max(meta.get("jobs", 1), 1)) if queued else base
            flag = " ⚠ after deadline" if deadline and batch > deadline else ""
            out += ["", f"Batch ETA: {clock(batch)}{flag} ({len(running)} running, {len(queued)} queued, jobs {meta.get('jobs', 1)})"]
    return "\n".join(out), not (running or queued)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iteration", type=Path, nargs="?", help="pipelines-N folder (default: the latest)")
    ap.add_argument("--condition", help="show one condition's cells")
    ap.add_argument("--deadline", help="HH:MM; flag conditions and the batch that would finish after it")
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="refresh every N seconds until the batch finishes")
    args = ap.parse_args()
    spec = matrix.load_spec()
    iteration = args.iteration or matrix.latest_iteration()
    if not iteration or not iteration.exists():
        print("no pipelines-N batch found", file=sys.stderr)
        return 2
    deadline = parse_deadline(args.deadline)
    while True:
        text, all_done = render(iteration, spec, deadline, args.condition)
        print(("\033[2J\033[H" if args.watch else "") + text, flush=True)
        if not args.watch or all_done:
            return 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
