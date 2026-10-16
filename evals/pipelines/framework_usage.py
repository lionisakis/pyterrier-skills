"""Did the agent build each pipeline component with the PyTerrier stack, or write its own?

Static scan of the code the agent left in its condition folder (*.py, *.sh, *.ipynb; venvs, caches,
indexes, the harness's docs mirror and copied skills are skipped). Comments and string literals are
blanked first, so a docstring or report line naming a library does not count as using it. Each component gets regex hits for
"framework" (PyTerrier or a PyTerrier plugin) and "own" (a hand-written implementation of the same thing);
the verdict is framework / own / mixed / – (not found). Hits are evidence, not proof: read the listed
lines before quoting a verdict.
"""

from __future__ import annotations

import io
import json
import os
import re
import tokenize
from collections import Counter
from pathlib import Path

SKIP_DIRS = {"venv", ".venv", "claude-config", "pyterrier_home", "indexes", "third_party", ".claude",
             "pyterrier-docs", "caches", "models", "runs", "results", "controls", "__pycache__",
             "site-packages", ".ipynb_checkpoints", "node_modules", ".git"}
SKIP_FILES = {"search_docs.py"}  # put there by the harness for the docs conditions

# component -> (framework pattern, own-implementation pattern)
COMPONENTS = {
    "bm25": (r"pt\.terrier\.Retriever|\bBatchRetrieve\(|\bTerrierRetrieve\(|terrier\.Retriever\(|\.bm25\(|PisaIndex|pyterrier_pisa",
             r"class \w*BM25|\brank_bm25\b|\bbm25s\b|\(\s*k1\s*\+\s*1(\.0)?\s*\)"),
    "rm3": (r"rewrite\.RM3|\bpt\.rewrite\.\w+|from pyterrier(\.terrier)?\.rewrite import|\.rm3\(",
            r"class \w*(RM3|RelevanceModel)|def (rm3|relevance_model)\w*\("),
    "splade": (r"\bpyt_splade\b|\bpyterrier_splade\b|SpladeFactory|pyterrier_dr\.\w*Splade",
               r"AutoModelForMaskedLM|log1p\(\s*(torch\.)?(F\.|nn\.functional\.)?relu|class \w*Splade"),
    "dense": (r"FlexIndex|HgfBiEncoder|pyterrier_dr\.\w+|\bRetroMAE\.\w+\(|\b(Numpy|Torch)Index\(",
              r"class \w*(RetroMAE|Dense|BiEncoder)\w*|\bAutoModel\.from_pretrained"),
    "monot5": (r"\bpyterrier_t5\b|MonoT5ReRanker|DuoT5ReRanker",
               r"T5ForConditionalGeneration|AutoModelForSeq2SeqLM|class \w*Mono[Tt]5"),
    "colbert": (r"\bpyterrier_colbert\b|ColbertPRF|ColBERTFactory|\bpylate\b|pyterrier_pylate",
                r"(?i)class \w*colbert|^\s*(from|import) colbert\b|\bmax_?sim\b|\.max\((dim=)?2\)\.values"),
    "composition": (r"\bpt\.apply\.\w+|\bpt\.Transformer\b|\bpt\.text\.get_text|>>\s*pt\.", None),
    "evaluation": (r"\bpt\.Experiment\(|\bpt\.Evaluate\(",
                   r"def \w*(ndcg|dcg|mrr|reciprocal_rank|average_precision)\w*\("),
    "datasets": (r"\bpt\.get_dataset\(|\bpt\.datasets\.|[\"']irds:", None),
}
ALIAS = re.compile(r"^\s*import (pyterrier|pyterrier_\w+|pyt_\w+) as (\w+)", re.M)
PT_API = re.compile(r"\bpt\.[a-z_]+(?:\.[A-Za-z_]+)*|\bimport (pyterrier_\w+|pyt_\w+)|\bfrom (pyterrier_\w+|pyt_\w+)")


def code_files(run_dir: Path) -> list[Path]:
    out = []
    for root, dirs, names in os.walk(run_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)  # prune before descending: venvs and indexes are huge
        out += [Path(root) / n for n in sorted(names)
                if n not in SKIP_FILES and n.endswith((".py", ".sh", ".ipynb"))]
    return out


def strip_python(text: str) -> str | None:
    """Python source with comments and string literals blanked, so docstrings and report text that name a
    library do not count as using it. Line numbers are kept. None when the source does not tokenize."""
    lines = text.splitlines(keepends=True)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING) or tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
                (r0, c0), (r1, c1) = tok.start, tok.end
                for r in range(r0, r1 + 1):
                    line = lines[r - 1]
                    a, b = (c0 if r == r0 else 0), (c1 if r == r1 else len(line.rstrip("\r\n")))
                    lines[r - 1] = line[:a] + " " * (b - a) + line[b:]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return "".join(lines)


def code_lines(path: Path) -> list[tuple[int, str]]:
    """Code lines without comments or string literals (notebooks: code cells only)."""
    try:
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            text = "\n".join("".join(c.get("source", "")) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return []
    if path.suffix != ".sh":
        text = strip_python(text) or text
    for module, alias in ALIAS.findall(text):  # `import pyterrier_dr as ptdr`: match ptdr.X as pyterrier_dr.X
        if alias != module:
            text = re.sub(rf"\b{alias}\.", ("pt" if module == "pyterrier" else module) + ".", text)
    return [(i, line) for i, line in enumerate(text.splitlines(), 1) if not line.lstrip().startswith("#")]


def scan(run_dir: Path) -> dict:
    files = code_files(run_dir)
    hits = {c: {"framework": [], "own": []} for c in COMPONENTS}
    api = Counter()
    pats = {c: (re.compile(f), re.compile(o) if o else None) for c, (f, o) in COMPONENTS.items()}
    for f in files:
        rel = str(f.relative_to(run_dir))
        for n, line in code_lines(f):
            for c, (fw, own) in pats.items():
                if fw.search(line):
                    hits[c]["framework"].append(f"{rel}:{n}")
                if own and own.search(line):
                    hits[c]["own"].append(f"{rel}:{n}")
            for m in PT_API.finditer(line):
                api[m.group(1) or m.group(2) or m.group(0)] += 1
    comps = {}
    for c, h in hits.items():
        fw, own = len(h["framework"]), len(h["own"])
        verdict = "mixed" if fw and own else "framework" if fw else "own" if own else "–"
        comps[c] = {"verdict": verdict, "framework": fw, "own": own,
                    "evidence": {"framework": h["framework"][:5], "own": h["own"][:5]}}
    judged = [v["verdict"] for v in comps.values() if v["verdict"] != "–"]
    return {"files": len(files), "components": comps, "pt_api": dict(api.most_common(25)),
            "framework_share": round(sum(v == "framework" for v in judged) / len(judged), 2) if judged else None}


def markdown(usage: dict) -> list[str]:
    conds = list(usage)
    lines = ["## Framework usage", "",
             "Static scan of each condition's own code: did it build the component with PyTerrier / a plugin "
             "(framework), hand-write it (own), or both (mixed)? Counts are matching lines (framework/own). "
             "Evidence, not proof: check the lines in grade.json before quoting a verdict.", "",
             "| component | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for comp in COMPONENTS:
        cells = []
        for c in conds:
            v = usage[c]["components"][comp]
            cells.append(v["verdict"] if v["verdict"] == "–" else f"{v['verdict']} ({v['framework']}/{v['own']})")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")
    lines.append("| files scanned | " + " | ".join(str(usage[c]["files"]) for c in conds) + " |")
    lines.append("| framework share | " + " | ".join(
        f"{usage[c]['framework_share']:.0%}" if usage[c]["framework_share"] is not None else "–" for c in conds) + " |")
    lines += ["", "PyTerrier APIs and plugins referenced (top 8):", ""]
    for c in conds:
        top = ", ".join(f"`{k}`×{n}" for k, n in list(usage[c]["pt_api"].items())[:8])
        lines.append(f"- **{c}**: {top or '–'}")
    return lines + [""]
