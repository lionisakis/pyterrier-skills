#!/usr/bin/env python3
"""Grade a pipeline-matrix batch from the run files alone, whatever the agents claimed.

For every condition and dataset (each NanoBEIR subset separately) it loads the official qrels with
ir_datasets and runs one pt.Experiment over the condition's run files, with that condition's BM25 run
as baseline, a paired t-test and correction="bonferroni". Measures come from pipelines.json (RR@10 for
MS MARCO dev; graded nDCG@10 and rel>=2 AP/recall for TREC DL; nDCG@10 for NanoBEIR).

Same results: for each cell, the spread of the main metric across conditions, and against the reference
condition ("all"): the difference, a paired t-test p on per-query values, and the mean per-query
top-10 overlap and Kendall tau-b over the documents both top-10s share.
BM25 parity: every condition's BM25 should be within the tolerance of the reference's.
Framework usage: a static scan of each condition's own code, per component (BM25, RM3, SPLADE, dense,
monoT5, ColBERT, composition, evaluation, datasets): built with PyTerrier / a plugin, hand-written, or both
(pipelines/framework_usage.py). --usage-only prints just that table, without loading qrels or runs.

--controls also runs the harness-side control `bm25_ce` (each condition's BM25 top-100 re-ranked by
the plain cross-encoder, no score injection), cached under <batch>/controls/. It needs torch and
sentence-transformers and is slow on MS MARCO dev without a GPU.

Writes grade.json and grade.md into the batch folder.

Examples
    python evals/grade_pipelines.py
    python evals/grade_pipelines.py ../pyterrier-skills-evals/pipelines-1 --controls
    python evals/grade_pipelines.py ../pyterrier-skills-evals/pipelines-4 --usage-only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipelines"))
import framework_usage  # noqa: E402
import matrix  # noqa: E402

import ir_measures  # noqa: E402
import pandas as pd  # noqa: E402
import pyterrier as pt  # noqa: E402
from scipy import stats  # noqa: E402


def load_run(path: Path) -> pd.DataFrame:
    df = pt.io.read_results(str(path))
    df["qid"], df["docno"] = df["qid"].astype(str), df["docno"].astype(str)
    return df


def per_query(run: pd.DataFrame, qrels: pd.DataFrame, measure) -> dict:
    r = run.rename(columns={"qid": "query_id", "docno": "doc_id"})[["query_id", "doc_id", "score"]]
    q = qrels.rename(columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"})[["query_id", "doc_id", "relevance"]]
    return {m.query_id: m.value for m in ir_measures.iter_calc([measure], q, r)}


def top10_agreement(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float | None, float | None]:
    """Mean top-10 overlap, and mean Kendall tau-b over the documents both top-10s share (>= 2 shared)."""
    overlaps, taus = [], []
    ga, gb = dict(tuple(a.groupby("qid"))), dict(tuple(b.groupby("qid")))
    for qid in set(ga) & set(gb):
        ta = ga[qid].sort_values("score", ascending=False)["docno"].head(10).tolist()
        tb = gb[qid].sort_values("score", ascending=False)["docno"].head(10).tolist()
        shared = [d for d in ta if d in tb]
        overlaps.append(len(shared) / max(len(ta), len(tb), 1))
        if len(shared) >= 2:
            tau = stats.kendalltau([ta.index(d) for d in shared], [tb.index(d) for d in shared]).statistic
            if not math.isnan(tau):
                taus.append(tau)
    mean = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731
    return mean(overlaps), mean(taus)


def bm25_ce_control(run_dir: Path, batch: Path, cell: dict, model_cache: dict) -> Path | None:
    """Re-rank the condition's BM25 top-100 with the plain cross-encoder (no score injection)."""
    out = batch / "controls" / run_dir.name / cell["slug"] / "bm25_ce.res.gz"
    if out.exists():
        return out
    src = matrix.find_run(run_dir, {**cell, "pipeline": "bm25"})
    if not src:
        return None
    if "ce" not in model_cache:
        from sentence_transformers import CrossEncoder
        model_cache["ce"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce = model_cache["ce"]
    ds = pt.get_dataset(f"irds:{cell['irds']}")
    topics = ds.get_topics()
    run = load_run(src)
    run = run.sort_values(["qid", "score"], ascending=[True, False]).groupby("qid").head(100)
    run = run.merge(topics[["qid", "query"]].astype({"qid": str}), on="qid")
    run = pt.text.get_text(ds, "text")(run)
    run["score"] = ce.predict(list(zip(run["query"], run["text"])), batch_size=64, show_progress_bar=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    pt.io.write_results(pt.model.add_ranks(run[["qid", "docno", "score"]]), str(out))
    return out


def grade(batch: Path, controls: bool, datasets: list[str] | None = None) -> dict:
    spec = matrix.load_spec()
    ref = spec["reference_condition"]
    dirs = matrix.condition_dirs(batch, spec)
    pipeline_ids = [p["id"] for p in spec["pipelines"]] + (["bm25_ce"] if controls else [])
    cells = matrix.all_cells(spec, datasets=datasets)
    keys = list(dict.fromkeys((c["column"], c["key"], c["irds"], c["slug"]) for c in cells))
    ds_spec = {d["id"]: d for d in spec["datasets"]}
    results: dict = {}      # condition -> key -> pipeline -> {metric: value, "sig": bool, "p": float}
    perq: dict = {}         # (condition, key, pipeline) -> {qid: main metric}
    runs: dict = {}
    model_cache: dict = {}
    for column, key, irds, slug in keys:
        d = ds_spec[column]
        measures = [ir_measures.parse_measure(m) for m in d["metrics"]]
        main = ir_measures.parse_measure(d["main"])
        ds = pt.get_dataset(f"irds:{irds}")
        qrels = ds.get_qrels().astype({"qid": str, "docno": str})
        for rd in dirs:
            cell = {"pipeline": "", "key": key, "irds": irds, "slug": slug, "column": column}
            found = {}
            for p in pipeline_ids:
                path = (bm25_ce_control(rd, batch, cell, model_cache) if p == "bm25_ce"
                        else matrix.find_run(rd, {**cell, "pipeline": p}))
                if path and matrix.check_run(path, {})["valid"]:
                    found[p] = load_run(path)
            if not found:
                continue
            names = list(found)
            baseline = names.index("bm25") if "bm25" in names and len(names) > 1 else None
            res = pt.Experiment([found[n] for n in names], ds.get_topics().astype({"qid": str}), qrels, measures,
                                names=names, baseline=baseline, correction="bonferroni" if baseline is not None else None)
            table = {}
            for _, row in res.iterrows():
                entry = {str(m): (round(float(row[str(m)]), 4) if pd.notna(row[str(m)]) else None) for m in measures}
                if baseline is not None and row["name"] != "bm25":
                    entry["p"] = float(row.get(f"{main} p-value corrected", row.get(f"{main} p-value", float("nan"))))
                    entry["sig"] = bool(row.get(f"{main} reject", False))
                    entry["better"] = entry["sig"] and entry[str(main)] > table.get("bm25", {}).get(str(main), entry[str(main)])
                table[row["name"]] = entry
            results.setdefault(rd.name, {})[key] = table
            for n in names:
                perq[(rd.name, key, n)] = per_query(found[n], qrels, main)
                runs[(rd.name, key, n)] = found[n].sort_values(["qid", "score"], ascending=[True, False]).groupby("qid").head(10)
            print(f"graded {rd.name:<12} {key:<28} {len(names)} runs", flush=True)

    agreement = []
    for column, key, irds, slug in keys:
        main = ds_spec[column]["main"]
        for p in pipeline_ids:
            vals = {c: results[c][key][p][main] for c in results if key in results[c] and p in results[c][key]}
            if not vals:
                continue
            row = {"pipeline": p, "key": key, "column": column, "n": len(vals), "min": min(vals.values()),
                   "max": max(vals.values()), "std": round(statistics.pstdev(vals.values()), 4), "vs_ref": {}}
            if (ref, key, p) in perq:
                rq = perq[(ref, key, p)]
                for c in vals:
                    if c == ref:
                        continue
                    cq = perq[(c, key, p)]
                    qids = sorted(set(rq) | set(cq))
                    a, b = [rq.get(q, 0.0) for q in qids], [cq.get(q, 0.0) for q in qids]
                    pval = stats.ttest_rel(a, b).pvalue if len(qids) > 1 and a != b else 1.0
                    overlap, tau = top10_agreement(runs[(ref, key, p)], runs[(c, key, p)])
                    row["vs_ref"][c] = {"delta": round(vals[c] - vals[ref], 4), "p": round(float(pval), 4),
                                        "overlap": overlap, "tau": tau}
            agreement.append(row)

    tol = spec["bm25_parity_tolerance"]
    parity = {}
    for c in results:
        diffs = [abs(results[c][k]["bm25"][ds_spec[col]["main"]] - results[ref][k]["bm25"][ds_spec[col]["main"]])
                 for col, k, _, _ in keys if k in results[c] and k in results.get(ref, {})
                 and "bm25" in results[c][k] and "bm25" in results[ref][k]]
        parity[c] = {"ok": all(x <= tol for x in diffs) if diffs else None, "max_diff": round(max(diffs), 4) if diffs else None}

    nano = next(d for d in spec["datasets"] if d.get("subsets"))
    nano_mean = {}
    for c in results if any(k.startswith("nano-beir/") for r in results.values() for k in r) else []:
        for p in pipeline_ids:
            vals = [results[c][s][p][nano["main"]] for s in nano["subsets"] if s in results[c] and p in results[c][s]]
            if vals:
                nano_mean.setdefault(c, {})[p] = {"mean": round(statistics.mean(vals), 4), "subsets": len(vals)}
    return {"reference": ref, "results": results, "agreement": agreement, "bm25_parity": parity,
            "nanobeir_mean": nano_mean, "pipelines": pipeline_ids, "framework_usage": usage(batch, spec)}


def usage(batch: Path, spec: dict) -> dict:
    return {rd.name: framework_usage.scan(rd) for rd in matrix.condition_dirs(batch, spec)}


def markdown(g: dict, spec: dict) -> str:
    conds = [c["id"] for c in spec["conditions"] if c["id"] in g["results"]]
    ref = g["reference"]
    lines = [f"# Grade: pipeline matrix", "",
             f"Main metric per dataset; † = significantly different from that condition's BM25 (paired t-test, "
             f"Bonferroni); (Δ) = difference from `{ref}`.", ""]
    for d in spec["datasets"]:
        for key in (d.get("subsets") or [d["id"]]):
            if not any(key in g["results"][c] for c in conds):
                continue
            lines += [f"## {key} — {d['main']}", "", "| pipeline | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
            for p in g["pipelines"]:
                cells = []
                for c in conds:
                    e = g["results"][c].get(key, {}).get(p)
                    if not e:
                        cells.append("–")
                        continue
                    v = e[d["main"]]
                    r = g["results"].get(ref, {}).get(key, {}).get(p, {}).get(d["main"])
                    cells.append(f"{v:.3f}{'†' if e.get('sig') else ''}" + (f" ({v - r:+.3f})" if r is not None and c != ref else ""))
                lines.append(f"| {p} | " + " | ".join(cells) + " |")
            lines.append("")
        if d.get("subsets"):
            lines += [f"## NanoBEIR mean over sub-datasets — {d['main']}", "",
                      "| pipeline | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
            for p in g["pipelines"]:
                lines.append(f"| {p} | " + " | ".join(
                    (f"{m['mean']:.3f}" + (f" ({m['subsets']}/13)" if m["subsets"] < 13 else "")) if (m := g["nanobeir_mean"].get(c, {}).get(p)) else "–"
                    for c in conds) + " |")
            lines.append("")
    lines += ["## All measures", ""]
    for c in conds:
        for key, table in g["results"][c].items():
            ms = [m for m in next(iter(table.values())) if m not in ("p", "sig", "better")]
            lines += [f"**{c} · {key}**", "", "| pipeline | " + " | ".join(ms) + " | p (corr.) |", "|---|" + "---|" * (len(ms) + 1)]
            for p, e in table.items():
                vals = " | ".join(f"{e[m]:.4f}" if e[m] is not None else "–" for m in ms)
                sig = f"{e['p']:.4f}" + ("†" if e.get("sig") else "") if "p" in e else "baseline"
                lines.append(f"| {p} | {vals} | {sig} |")
            lines.append("")
    lines += ["## Agreement across conditions", "",
              f"Spread of the main metric over conditions; vs `{ref}`: Δ, paired t-test p on per-query values, "
              "mean top-10 overlap, mean Kendall τ over the shared top-10 documents.", "",
              "| pipeline | dataset | n | min | max | std | vs ref (Δ / p / overlap / τ) |", "|---|---|---|---|---|---|---|"]
    for a in g["agreement"]:
        vs = "; ".join(f"{c}: {v['delta']:+.3f} / {v['p']:.3f} / {v['overlap'] if v['overlap'] is not None else '–'} / {v['tau'] if v['tau'] is not None else '–'}"
                       for c, v in a["vs_ref"].items())
        lines.append(f"| {a['pipeline']} | {a['key']} | {a['n']} | {a['min']:.3f} | {a['max']:.3f} | {a['std']:.3f} | {vs or '–'} |")
    lines += ["", "## BM25 parity", "", f"Every condition's BM25 vs `{ref}`'s, tolerance {spec['bm25_parity_tolerance']}.", "",
              "| condition | ok | max diff |", "|---|---|---|"]
    for c, v in g["bm25_parity"].items():
        lines.append(f"| {c} | {'✓' if v['ok'] else '✗' if v['ok'] is False else '–'} | {v['max_diff'] if v['max_diff'] is not None else '–'} |")
    if g.get("framework_usage"):
        lines += [""] + framework_usage.markdown(g["framework_usage"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", type=Path, nargs="?", help="pipelines-N folder (default: the latest)")
    ap.add_argument("--controls", action="store_true", help="also run the bm25_ce control (cross-encoder, slow)")
    ap.add_argument("--datasets", help="comma-separated dataset ids to grade (default: all)")
    ap.add_argument("--usage-only", action="store_true", help="print only the framework-usage table (fast; writes nothing)")
    args = ap.parse_args()
    batch = args.batch or matrix.latest_iteration()
    if not batch:
        print("no pipelines-N batch found", file=sys.stderr)
        return 2
    spec = matrix.load_spec()
    if args.usage_only:
        print("\n".join(framework_usage.markdown(usage(batch, spec))))
        return 0
    g = grade(batch, args.controls, args.datasets.split(",") if args.datasets else None)
    (batch / "grade.json").write_text(json.dumps(g, indent=2, default=str), encoding="utf-8")
    (batch / "grade.md").write_text(markdown(g, spec), encoding="utf-8")
    print(f"grade: {batch / 'grade.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
