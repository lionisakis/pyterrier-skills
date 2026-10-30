#!/usr/bin/env python3
"""Mirror the PyTerrier documentation (core + every ext/ page) as text, for the pipeline-matrix evals.

Crawls https://pyterrier.readthedocs.io/en/latest/ (pages under that prefix only). For each page it
saves the Sphinx source that the page links to (`_sources/<page>.rst.txt` / `.md.txt` / `.ipynb.txt`),
which is plain text with code blocks intact; pages without a source link are saved as text extracted
from the HTML. Output goes to `<repo>/pyterrier-docs/` (gitignored), one file per page, plus
`index.json` mapping each file to its URL and title. `search_docs.py` retrieves from it.

Examples
    python evals/fetch_docs.py
    python evals/fetch_docs.py --out /tmp/docs --max-pages 50
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = "https://pyterrier.readthedocs.io/en/latest/"
HREF = re.compile(r'href="([^"#?]+)')
TITLE = re.compile(r"<title>(.*?)</title>", re.S)


def get(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "pyterrier-skills-evals/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2 * (attempt + 1))
    return None


def html_to_text(page: str) -> str:
    main = re.search(r'<article[^>]*role="main"[^>]*>(.*?)</article>', page, re.S) or re.search(r"<body.*?>(.*)</body>", page, re.S)
    body = main.group(1) if main else page
    body = re.sub(r"<(script|style|nav|footer)\b.*?</\1>", "", body, flags=re.S)
    body = re.sub(r"<pre[^>]*>(.*?)</pre>", lambda m: "\n```\n" + re.sub(r"<[^>]+>", "", m.group(1)) + "\n```\n", body, flags=re.S)
    body = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "\n" + "#" * int(m.group(1)) + " " + m.group(2) + "\n", body, flags=re.S)
    body = re.sub(r"<(br|/p|/li|/tr|/div)[^>]*>", "\n", body)
    text = html.unescape(re.sub(r"<[^>]+>", "", body)).replace("¶", "")
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip() + "\n"


def crawl(out: Path, max_pages: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    queue, seen, index = ["index.html"], {"index.html"}, {}
    while queue and len(index) < max_pages:
        rel = queue.pop(0)
        page = get(BASE + rel)
        if page is None:
            continue
        for href in HREF.findall(page):
            target = urllib.parse.urljoin(BASE + rel, href)
            if not target.startswith(BASE) or not target.endswith(".html"):
                continue
            t = target[len(BASE):]
            if t.startswith(("_", "genindex", "search", "py-modindex")) or t in seen:
                continue
            seen.add(t)
            queue.append(t)
        src = next((urllib.parse.urljoin(BASE + rel, h) for h in HREF.findall(page) if "/_sources/" in urllib.parse.urljoin(BASE + rel, h)), None)
        text = get(src) if src else None
        kind = "source" if text else "html"
        text = text or html_to_text(page)
        name = rel[:-len(".html")].replace("/", "__") + ".txt"
        title = html.unescape(TITLE.search(page).group(1)).split(" - ")[0].strip() if TITLE.search(page) else rel
        (out / name).write_text(f"# {title}\nURL: {BASE + rel}\n\n{text}", encoding="utf-8")
        index[name] = {"url": BASE + rel, "title": title, "from": kind}
        print(f"{len(index):>4}  {kind:<6} {rel}", flush=True)
        time.sleep(0.2)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return len(index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "pyterrier-docs")
    ap.add_argument("--max-pages", type=int, default=2000)
    args = ap.parse_args()
    n = crawl(args.out, args.max_pages)
    print(f"{n} pages -> {args.out}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
