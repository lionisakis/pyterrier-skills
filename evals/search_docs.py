#!/usr/bin/env python3
"""Search the local PyTerrier documentation mirror (pyterrier-docs/) with BM25. Standard library only.

Each page is split into sections at Markdown/RST headings; sections are ranked with BM25 and the top
ones are printed as `file:line  title > heading` with a snippet. Open the file to read more.

Examples
    python search_docs.py "splade pisa index"
    python search_docs.py "monoT5 reranker" -k 5 --full
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = next((p for p in (HERE / "pyterrier-docs", HERE.parent / "pyterrier-docs") if p.is_dir()), HERE / "pyterrier-docs")
TOKEN = re.compile(r"[a-z0-9_]+")
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


def sections(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    title = lines[0].lstrip("# ").strip() if lines else path.stem
    start, heading = 0, title
    for i, line in enumerate(lines):
        new = None
        if line.startswith("#") and i > 0:
            new = line.lstrip("#").strip()
        elif i > 0 and RST_UNDERLINE.match(line) and lines[i - 1].strip():
            new = lines[i - 1].strip()
            i -= 1
        if new and i > start:
            yield start, title, heading, lines[start:i]
            start, heading = i, new
    yield start, title, heading, lines[start:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8, help="results to show")
    ap.add_argument("--full", action="store_true", help="print whole sections instead of snippets")
    args = ap.parse_args()
    if not DOCS.is_dir():
        print(f"no docs at {DOCS}", file=sys.stderr)
        return 1
    secs = [(f, *s) for f in sorted(DOCS.glob("*.txt")) for s in sections(f)]
    tfs = [Counter(TOKEN.findall((s[3] + " " + "\n".join(s[4])).lower())) for s in secs]
    df = Counter(t for tf in tfs for t in tf)
    n, avg = len(secs), sum(sum(tf.values()) for tf in tfs) / max(len(secs), 1)
    q = TOKEN.findall(args.query.lower())
    scores = []
    for i, tf in enumerate(tfs):
        dl, s = sum(tf.values()), 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * dl / avg))
        if s:
            scores.append((s, i))
    for s, i in sorted(scores, reverse=True)[:args.k]:
        f, start, title, heading, body = secs[i]
        print(f"\n== {f.name}:{start + 1}  {title} > {heading}  (score {s:.1f})")
        text = [l for l in body if l.strip()]
        if args.full:
            print("\n".join(body))
        else:
            hits = [j for j, l in enumerate(text) if any(t in l.lower() for t in q)] or [0]
            print("\n".join(text[max(hits[0] - 2, 0):hits[0] + 10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
