#!/usr/bin/env python3
"""Count how often eval agents read library source instead of the skills.

A skill should be enough to do the task. Every time an agent opens installed source (site-packages),
calls inspect/help, or greps a package, it needed something the skill did not say. This script reads
each run's transcript (claude-config/projects/**/*.jsonl, present for runs that used --env-file with
an API key) and counts, per eval:

    skill reads   Read/cat/sed/grep of files under .claude/skills/
    source reads  the same on site-packages, plus inspect.getsource/signature, help(), __file__ probes
    by package    which packages the source reads were in

Examples
    python evals/source_reads.py                      # latest iteration
    python evals/source_reads.py ../pyterrier-skills-evals/iteration-4 --details
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = REPO.parent / f"{REPO.name}-evals"

SITE = re.compile(r"site-packages/([A-Za-z0-9_\-]+)")
PROBE = re.compile(r"inspect\.(getsource|signature|getfile|getmembers)|\bhelp\(|__file__|\bdir\(")
IMPORT = re.compile(r"(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)")
SKILL = re.compile(r"\.claude/skills/([a-z0-9\-]+)")


def tool_calls(transcript: Path):
    for line in transcript.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        content = row.get("message", {}).get("content")
        if row.get("type") != "assistant" or not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use":
                inp = c["input"]
                yield c["name"], str(inp.get("command") or inp.get("file_path") or inp.get("path") or inp.get("pattern") or "")


def classify(name: str, text: str):
    """Return ("skill", skill) / ("source", package) / None for one tool call."""
    if m := SKILL.search(text):
        return "skill", m.group(1)
    if m := SITE.search(text):
        return "source", m.group(1).split("-")[0]
    if name == "Bash" and PROBE.search(text):
        mods = [m for m in IMPORT.findall(text) if m not in sys.stdlib_module_names]
        return "source", (mods[0] if mods else "?") + " (inspect/help)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("iteration", type=Path, nargs="?", help="iteration-N folder (default: the latest)")
    ap.add_argument("--details", action="store_true", help="list every source read")
    args = ap.parse_args()
    iteration = args.iteration or max(DEFAULT_WORKSPACE.glob("iteration-*"),
                                      key=lambda p: int(p.name.split("-")[1]))
    print(f"{iteration}\n")
    print(f"{'eval':<32} {'calls':>5} {'skill':>5} {'source':>6}  top packages read")
    for run in sorted(iteration.glob("eval-*"), key=lambda p: int(p.name.split("-")[1])):
        transcripts = list(run.glob("claude-config/projects/**/*.jsonl"))
        if not transcripts:
            print(f"{run.name:<32} no transcript (run without --env-file/API key?)")
            continue
        calls, skills, sources, details = 0, Counter(), Counter(), []
        for t in transcripts:
            for name, text in tool_calls(t):
                calls += 1
                kind = classify(name, text)
                if kind and kind[0] == "skill":
                    skills[kind[1]] += 1
                elif kind:
                    sources[kind[1]] += 1
                    details.append((kind[1], text.replace("\n", " ")[:140]))
        top = ", ".join(f"{p} {n}" for p, n in sources.most_common(4))
        print(f"{run.name:<32} {calls:>5} {sum(skills.values()):>5} {sum(sources.values()):>6}  {top}")
        if args.details:
            for pkg, text in details:
                print(f"    [{pkg}] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
