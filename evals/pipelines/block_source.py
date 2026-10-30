#!/usr/bin/env python3
"""PreToolUse hook for the pipeline-matrix evals: enforce what a condition may look at.

run_pipelines.py writes `<run_dir>/hook_config.json` and registers this script in the project's
`.claude/settings.json` as `python3 block_source.py <run_dir>/hook_config.json`. Claude Code sends the
tool call as JSON on stdin; exit code 2 blocks it and the stderr text is shown to the agent.

Always blocked
    paths in this repo (the skills, the evals, other docs) and in the other conditions' folders
Blocked without code view
    reading the agent's venv or any site-packages (Read/Grep/Glob/Bash paths), pip show --files,
    inspect / help() / pydoc / __file__ / __doc__ probes (in Bash or in written scripts),
    git clone and curl/wget of GitHub, WebFetch of GitHub
Blocked without docs
    the docs mirror, WebFetch/WebSearch, curl/wget of readthedocs

The checks are string and path based, so a determined agent can get round them; every block is
logged to `<run_dir>/hook_blocks.jsonl`, and source_reads.py counts any reads that slip through.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone

PROBE = re.compile(r"inspect\.(getsource|getsourcefile|getfile|signature|getmembers|getdoc)|\binspect\b.*import|"
                   r"import\s+inspect|\bhelp\(|\bpydoc\b|__file__|__doc__|\bpip3?\s+show\s+(-f|--files)")
SITE = re.compile(r"site-packages|dist-packages|/lib/python3[\d.]*/")
GITHUB = re.compile(r"github\.com|githubusercontent\.com|gitlab\.com|sourcegraph\.com")
FETCH = re.compile(r"\b(curl|wget|git\s+clone|gh\s+repo|svn\s+(co|checkout))\b")
DOCS_SITES = re.compile(r"readthedocs\.(io|org)|pyterrier\.org")


def inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def paths_in(text: str, cwd: str) -> list[str]:
    """Absolute, resolved forms of every path-like token in a command or argument."""
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    out = []
    for tok in tokens:
        for part in re.split(r"[=:;,()<>|&\"']", tok):
            if "/" in part or part in ("..", "."):
                out.append(os.path.realpath(os.path.join(cwd, os.path.expanduser(part))))
    return out


def check(cfg: dict, tool: str, inp: dict, cwd: str) -> str | None:
    run_dir, iteration = cfg["run_dir"], cfg["iteration"]
    allowed_roots = [run_dir] + ([cfg["docs_dir"]] if cfg["docs"] else []) + cfg.get("allow", [])
    if tool in ("Read", "Grep", "Glob", "LS", "NotebookRead"):
        text = " ".join(str(inp.get(k, "")) for k in ("file_path", "path", "pattern", "glob"))
    elif tool == "Bash":
        text = str(inp.get("command", ""))
    elif tool in ("Write", "Edit", "MultiEdit"):
        text = str(inp.get("file_path", ""))
        body = str(inp.get("content", "")) + str(inp.get("new_string", ""))
        if not cfg["code"] and PROBE.search(body):
            return "No code view in this run: scripts may not inspect library source (inspect/help/__file__/pydoc)."
    elif tool in ("WebFetch", "WebSearch"):
        target = str(inp.get("url", "")) + " " + str(inp.get("query", ""))
        if not cfg["docs"]:
            return "No documentation in this run: web access is disabled."
        if not cfg["code"] and GITHUB.search(target):
            return "No code view in this run: source repositories (GitHub etc.) are not available."
        return None
    else:
        return None

    for p in paths_in(text, cwd):
        if inside(p, cfg["repo"]) and not any(inside(p, a) for a in allowed_roots):
            return "That path is outside this task's workspace."
        if inside(p, iteration) and not inside(p, run_dir):
            return "That path is outside this task's workspace."
        if not cfg["docs"] and inside(p, cfg["docs_dir"]):
            return "No documentation in this run."
        if not cfg["code"] and inside(p, os.path.join(run_dir, "venv")) and not inside(p, os.path.join(run_dir, "venv", "bin")):
            return "No code view in this run: the installed packages' files cannot be read."
    if not cfg["code"] and tool != "Bash" and SITE.search(text):
        return "No code view in this run: the installed packages' files cannot be read."
    if tool == "Bash":
        if not cfg["code"]:
            if SITE.search(text) and not re.search(r"\bpip3?\s+install\b", text):
                return "No code view in this run: the installed packages' files cannot be read."
            if PROBE.search(text):
                return "No code view in this run: inspect/help/__file__/pydoc/pip show --files are disabled."
            if GITHUB.search(text) and FETCH.search(text):
                return "No code view in this run: downloading or cloning source repositories is disabled (pip install is fine)."
        if not cfg["docs"] and DOCS_SITES.search(text) and FETCH.search(text):
            return "No documentation in this run: fetching documentation sites is disabled."
    return None


def main() -> int:
    cfg = json.loads(open(sys.argv[1], encoding="utf-8").read())
    event = json.loads(sys.stdin.read() or "{}")
    tool, inp = event.get("tool_name", ""), event.get("tool_input") or {}
    reason = check(cfg, tool, inp, event.get("cwd") or os.getcwd())
    if reason is None:
        return 0
    with open(os.path.join(cfg["run_dir"], "hook_blocks.jsonl"), "a", encoding="utf-8") as log:
        log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "tool": tool,
                              "input": {k: str(v)[:300] for k, v in inp.items()}, "reason": reason}) + "\n")
    print(reason, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
