#!/usr/bin/env python3
"""Report for a pipeline-matrix batch: one row per condition, so the 8 resource conditions compare at a glance.

Builds on report.eval_report() (time, interaction, tokens, CPU, reuse, skill/source reads, rigour) and
adds: cells done, outcome, env setup and index build times (manifest.json), time to the first valid
run (BM25 on DL19), tool errors, context compactions, hook blocks, tokens per finished cell, GPU use
(sampled by run_pipelines.py), index disk use, redundant index builds, prebuilt downloads (against the
rules), doc reads, web calls, source-read leaks, SuiteEval use, installed versions and model
substitutions, framework usage (pipelines/framework_usage.py: PyTerrier stack or hand-written, per
component). Quality columns come from grade.json (run grade_pipelines.py first). A cell time table
(minutes from first active to done, from status_log.jsonl) closes the report.

Writes report.md and report.json into the batch folder.

Examples
    python evals/pipeline_report.py
    python evals/pipeline_report.py ../pyterrier-skills-evals/pipelines-1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "pipelines"))
import framework_usage  # noqa: E402
import matrix  # noqa: E402
from report import agent_code, eval_report, fmt_s, read_rows  # noqa: E402

VERSIONED = re.compile(r"^(python-terrier|pyterrier[\w-]*|pyt[\w-]*|ir[-_]datasets|ir[-_]measures|suiteeval|torch|transformers|"
                       r"sentence-transformers|faiss[\w-]*|colbert[\w-]*)==", re.I)


def tool_results(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        content = r.get("message", {}).get("content")
        if isinstance(content, list):
            out += [c for c in content if c.get("type") == "tool_result"]
    return out


def tool_uses(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        content = r.get("message", {}).get("content")
        if r.get("type") == "assistant" and isinstance(content, list):
            out += [c for c in content if c.get("type") == "tool_use"]
    return out


def versions(run_dir: Path) -> dict:
    py = run_dir / "venv" / "bin" / "python"
    if not py.exists():
        return {}
    out = subprocess.run([str(py), "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
    return dict(l.split("==", 1) for l in out.splitlines() if VERSIONED.match(l))


def condition_report(rd: Path, spec: dict, cond: dict, grade: dict | None) -> dict:
    base = eval_report(rd, None)
    rows = read_rows(rd)
    cells = matrix.condition_cells(rd, spec)
    states = matrix.cell_states(rd, cells, {}, matrix.expected_queries(spec))
    st = matrix.read_json(rd / "agent_state.json", {}) or {}
    timing = matrix.read_json(rd / "timing.json", {}) or {}
    manifest = matrix.read_json(rd / "manifest.json", {}) or {}
    uses = tool_uses(rows)
    code = "\n".join(agent_code(rows))
    done = sum(s == "done" for s in states.values())

    indexes = [i for i in manifest.get("indexes", []) if isinstance(i, dict)]
    by_kind = Counter((str(i.get("type")), str(i.get("corpus"))) for i in indexes)
    started = st.get("started")
    first = next((e["time"] for e in matrix.status_log(rd) if e["pipeline"] == "bm25" and e["key"] == "dl19" and e["state"] == "done"), None)
    spec_models = {p["id"]: set(p["models"]) for p in spec["pipelines"]}
    substituted = sorted(p for p, v in (manifest.get("pipelines") or {}).items()
                         if isinstance(v, dict) and p in spec_models and spec_models[p]
                         and not all(any(m in str(x) for x in v.get("models", [])) for m in spec_models[p] if "/" in m))
    blocks = (rd / "hook_blocks.jsonl").read_text(encoding="utf-8").count("\n") if (rd / "hook_blocks.jsonl").exists() else 0
    bash = [str(u["input"].get("command", "")) for u in uses if u["name"] == "Bash"]
    doc_reads = sum("search_docs.py" in c or "pyterrier-docs" in c for c in bash) + sum(
        "pyterrier-docs" in str(u["input"].get("file_path") or u["input"].get("path") or "") for u in uses if u["name"] in ("Read", "Grep", "Glob"))

    quality = {}
    if grade and rd.name in grade["results"]:
        res = grade["results"][rd.name]
        ds = {d["id"]: d for d in spec["datasets"]}
        col_of = {c["key"]: c["column"] for c in matrix.all_cells(spec)}
        ndcg = [e.get("nDCG@10") for k, t in res.items() for p, e in t.items() if p != "bm25_ce" and e.get("nDCG@10") is not None]
        better = {k: sum(bool(e.get("better")) for p, e in t.items() if p != "bm25_ce") for k, t in res.items()}
        vs = [a["vs_ref"][rd.name] for a in grade["agreement"] if rd.name in a["vs_ref"]]
        quality = {
            "mean_ndcg10": round(statistics.mean(ndcg), 4) if ndcg else None,
            "sig_better_than_bm25": {k: better[k] for k in better if not k.startswith("nano-beir/")}
            | ({"nanobeir (sum)": sum(v for k, v in better.items() if k.startswith("nano-beir/"))} if any(k.startswith("nano-beir/") for k in better) else {}),
            "bm25_parity": grade["bm25_parity"].get(rd.name, {}).get("ok"),
            "mean_abs_delta_vs_ref": round(statistics.mean(abs(v["delta"]) for v in vs), 4) if vs else None,
            "mean_overlap_vs_ref": round(statistics.mean(v["overlap"] for v in vs if v.get("overlap") is not None), 3) if any(v.get("overlap") is not None for v in vs) else None,
            "mean_tau_vs_ref": round(statistics.mean(v["tau"] for v in vs if v["tau"] is not None), 3) if any(v["tau"] is not None for v in vs) else None,
            "cells_off_vs_ref": sum(abs(v["delta"]) > spec["cell_off_threshold"] for v in vs),
            "main_metric_by_column": {col_of.get(k, k): ds[col_of.get(k, k)]["main"] for k in res if col_of.get(k) in ds},
        }

    if timing:
        base["interaction"]["human_messages"] = 1 + (timing.get("nudges") or 0)
    return {
        **base,
        "condition": rd.name, "flags": {k: cond[k] for k in ("skills", "docs", "code")},
        "outcome": st.get("outcome") or st.get("state", "queued"), "cells_done": done, "cells": len(cells),
        "cells_failed": sum(s == "failed" for s in states.values()),
        "hit_max_turns": "error_max_turns" in (matrix.read_json(rd / "result.json", {}) or {}).get("subtypes", []),
        "build": {"env_setup_s": manifest.get("env_setup_s"),
                  "index_build_s": {f"{i.get('type')}:{i.get('corpus')}": i.get("build_s") for i in indexes},
                  "first_valid_run_s": round((datetime.fromisoformat(first) - datetime.fromisoformat(started)).total_seconds())
                  if first and started else None},
        "extra_interaction": {"nudges": timing.get("nudges", 0), "api_retries": timing.get("api_retries", 0),
                              "tool_errors": sum(bool(r.get("is_error")) for r in tool_results(rows)),
                              "compactions": sum(r.get("type") == "system" and r.get("subtype") == "compact_boundary"
                                                 or bool(r.get("isCompactSummary")) for r in rows),
                              "hook_blocks": blocks},
        "tokens_per_cell": round(base["tokens"]["total"] / done) if done else None,
        "gpu": {k: timing.get(k) for k in ("gpu_seconds", "gpu_util_mean", "gpu_mem_peak_mb", "gpu")},
        "disk_gb": round(matrix.dir_size(rd / "indexes") / 1e9, 2) if (rd / "indexes").exists() else 0,
        "reuse_extra": {"indexes": {f"{t}:{c}": n for (t, c), n in by_kind.items()},
                        "redundant_builds": sum(n - 1 for n in by_kind.values()),
                        "prebuilt_downloads": base["reuse"]["prebuilt_loads"],
                        "non_hf_prebuilt": base["reuse"].get("non_hf_prebuilt_loads", 0)},
        "resources": {"doc_reads": doc_reads,
                      "web_calls": sum(u["name"] in ("WebFetch", "WebSearch") for u in uses),
                      "leaks": base["source"]["source_reads"] if not cond["code"] else 0,
                      "used_suiteeval": bool(re.search(r"\bsuiteeval\b", code))},
        "quality": quality,
        "framework": framework_usage.scan(rd),
        "repro": {"versions": versions(rd), "substituted_models": substituted},
    }


def cell_time_table(spec: dict, dirs: list[Path]) -> list[str]:
    conds = [d.name for d in dirs]
    times = {d.name: matrix.cell_times(d) for d in dirs}
    lines = ["## Cell times (minutes from first active to done)", "", "| pipeline | dataset | " + " | ".join(conds) + " |",
             "|---|---|" + "---|" * len(conds)]
    for p in spec["pipelines"]:
        for d in spec["datasets"]:
            keys = d.get("subsets") or [d["id"]]
            vals = []
            for c in conds:
                mins = []
                for k in keys:
                    t = times[c].get((p["id"], k), {})
                    if t.get("first_active") and t.get("done"):
                        mins.append((datetime.fromisoformat(t["done"]) - datetime.fromisoformat(t["first_active"])).total_seconds() / 60)
                vals.append(f"{sum(mins):.0f}" + (f" ({len(mins)}/{len(keys)})" if len(keys) > 1 and len(mins) < len(keys) else "") if mins else "–")
            lines.append(f"| {p['id']} | {d['id']} | " + " | ".join(vals) + " |")
    return lines + ["", "Cells that finish between two 30 s polls without a building/running state show as –."]


def markdown(batch: Path, spec: dict, reps: list[dict], meta: dict, graded: bool) -> str:
    yn = lambda b: "✓" if b else "–"  # noqa: E731
    lines = [f"# Pipeline-matrix report: {batch.name}", "",
             f"- Model: {meta.get('model', '?')} · effort: {meta.get('effort') or 'default'} · max turns {meta.get('max_turns')} · "
             f"nudges {meta.get('max_nudges')} · GPU: {meta.get('gpu', '?')}",
             f"- Grade: {'from grade.json' if graded else 'not graded yet (run grade_pipelines.py)'}", "",
             "## Outcome and time", "",
             "| condition | S | D | C | outcome | cells | failed | wall | LLM | tools+code | env setup | first run | index builds (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        f, b = r["flags"], r["build"]
        idx = ", ".join(f"{k} {fmt_s(round(v)) if isinstance(v, (int, float)) else '?'}" for k, v in b["index_build_s"].items()) or "–"
        lines.append(f"| {r['condition']} | {yn(f['skills'])} | {yn(f['docs'])} | {yn(f['code'])} | {r['outcome']}"
                     f"{' (max turns)' if r['hit_max_turns'] else ''} | {r['cells_done']}/{r['cells']} | {r['cells_failed']} | "
                     f"{fmt_s(r['time']['wall_s'])} | {fmt_s(r['time']['llm_s'])} | {fmt_s(r['time']['tools_and_code_s'])} | "
                     f"{fmt_s(round(b['env_setup_s'])) if isinstance(b['env_setup_s'], (int, float)) else '–'} | "
                     f"{fmt_s(b['first_valid_run_s'])} | {idx} |")
    lines += ["", "## Interaction and tokens", "",
              "| condition | human msgs | nudges | API retries | LLM turns | tool calls | tool errors | compactions | hook blocks | "
              "input | cache write | cache read | output | total | tokens/cell | cost est. |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        i, x, t = r["interaction"], r["extra_interaction"], r["tokens"]
        lines.append(f"| {r['condition']} | {i['human_messages']} | {x['nudges']} | {x['api_retries']} | {i['llm_turns']} | {i['tool_calls']} | "
                     f"{x['tool_errors']} | {x['compactions']} | {x['hook_blocks']} | {t['input']:,} | {t['cache_write']:,} | "
                     f"{t['cache_read']:,} | {t['output']:,} | {t['total']:,} | "
                     f"{format(r['tokens_per_cell'], ',') if r['tokens_per_cell'] else '–'} | ${r['cost_estimate_usd'] or 0:.2f} |")
    lines += ["", "## Compute and reuse", "",
              "| condition | CPU | peak RSS | GPU-s | GPU util | GPU mem peak | index disk | indexes (type:corpus × builds) | "
              "redundant builds | caches | prebuilt downloads |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        c, g, x = r["compute"], r["gpu"], r["reuse_extra"]
        cpu = (c.get("cpu_user_s") or 0) + (c.get("cpu_system_s") or 0) if c.get("cpu_user_s") is not None else None
        lines.append(f"| {r['condition']} | {fmt_s(round(cpu)) if cpu is not None else '–'} | {c.get('max_rss_mb') or '–'} MB | "
                     f"{g.get('gpu_seconds') if g.get('gpu_seconds') is not None else '–'} | {g.get('gpu_util_mean') or '–'}% | "
                     f"{g.get('gpu_mem_peak_mb') or '–'} MB | {r['disk_gb']} GB | "
                     f"{', '.join(f'{k}×{v}' for k, v in x['indexes'].items()) or '–'} | {x['redundant_builds']} | "
                     f"{r['reuse']['cache_uses']} | {x['prebuilt_downloads']}{' (%d non-HF ⚠)' % x['non_hf_prebuilt'] if x['non_hf_prebuilt'] else ''} |")
    lines += ["", "## Resources used", "",
              "| condition | skill reads | doc reads | web calls | source reads | leaks | SuiteEval | Experiment: baseline · correction |",
              "|---|---|---|---|---|---|---|---|"]
    for r in reps:
        s, g = r["resources"], r["rigour"]
        rig = (f"{g['with_baseline']}/{g['calls']} · {', '.join(f'{k}×{v}' for k, v in g['corrections'].items()) or 'none'}"
               if g["calls"] else "no pt.Experiment")
        lines.append(f"| {r['condition']} | {r['source']['skill_reads']} | {s['doc_reads']} | {s['web_calls']} | "
                     f"{r['source']['source_reads']} | {s['leaks']}{' ⚠' if s['leaks'] else ''} | {yn(s['used_suiteeval'])} | {rig} |")
    lines += ["", *framework_usage.markdown({r["condition"]: r["framework"] for r in reps})]
    lines += ["## Quality (from grade.json)", "",
              "| condition | mean nDCG@10 | sig. better than BM25 (per dataset) | BM25 parity | mean \\|Δ\\| vs ref | top-10 overlap vs ref | mean τ vs ref | cells off |",
              "|---|---|---|---|---|---|---|---|"]
    for r in reps:
        q = r["quality"]
        if not q:
            lines.append(f"| {r['condition']} | – | – | – | – | – | – | – |")
            continue
        sig = ", ".join(f"{k}: {v}" for k, v in q["sig_better_than_bm25"].items()) or "–"
        par = "✓" if q["bm25_parity"] else "✗" if q["bm25_parity"] is False else "–"
        lines.append(f"| {r['condition']} | {q['mean_ndcg10'] if q['mean_ndcg10'] is not None else '–'} | {sig} | {par} | "
                     f"{q['mean_abs_delta_vs_ref'] if q['mean_abs_delta_vs_ref'] is not None else '–'} | "
                     f"{q['mean_overlap_vs_ref'] if q['mean_overlap_vs_ref'] is not None else '–'} | "
                     f"{q['mean_tau_vs_ref'] if q['mean_tau_vs_ref'] is not None else '–'} | {q['cells_off_vs_ref']} |")
    lines += ["", "## Reproducibility", "", "| condition | installed versions | substituted models |", "|---|---|---|"]
    for r in reps:
        v = r["repro"]
        lines.append(f"| {r['condition']} | {', '.join(f'{k} {x}' for k, x in sorted(v['versions'].items())) or '–'} | "
                     f"{', '.join(v['substituted_models']) or 'none'} |")
    lines += ["", *cell_time_table(spec, [Path(r["dir"]) for r in reps]), "", "How to read it:",
              "- S / D / C = skills / documentation / code view. human msgs = the prompt plus nudges.",
              "- first run = time from start to the first valid BM25 run on DL19.",
              "- leaks = library-source reads in a condition without code view (the hook missed them).",
              "- prebuilt downloads are allowed only from Hugging Face; ⚠ marks loads from anywhere else.",
              "- cells off = cells whose main metric differs from the reference condition by more than "
              f"{spec['cell_off_threshold']}.",
              "- grade.md has the per-dataset tables (significance vs BM25 with Bonferroni) and the agreement table."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", type=Path, nargs="?", help="pipelines-N folder (default: the latest)")
    args = ap.parse_args()
    batch = args.batch or matrix.latest_iteration()
    if not batch:
        print("no pipelines-N batch found", file=sys.stderr)
        return 2
    spec = matrix.load_spec()
    meta = matrix.read_json(batch / "run_meta.json", {}) or {}
    grade = matrix.read_json(batch / "grade.json")
    conds = {c["id"]: c for c in spec["conditions"]}
    reps = []
    for rd in matrix.condition_dirs(batch, spec):
        r = condition_report(rd, spec, conds[rd.name], grade)
        r["dir"] = str(rd)
        reps.append(r)
    (batch / "report.json").write_text(json.dumps({"meta": meta, "conditions": reps}, indent=2, default=str), encoding="utf-8")
    (batch / "report.md").write_text(markdown(batch, spec, reps, meta, grade is not None), encoding="utf-8")
    print((batch / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
