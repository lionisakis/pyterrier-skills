#!/usr/bin/env bash
# Runs the skill evals (evals/run_evals.py) in the foreground; output goes to the terminal and to
# logs/evals-<time>.log. Closing the terminal stops the run (Ctrl-C stops it too).
#
#   bash evals/run_evals.sh                       # all evals into a new iteration-N
#   FROM=iteration-14 bash evals/run_evals.sh     # only the evals that have no findings.md in iteration-14
#   IDS="4 5 6" bash evals/run_evals.sh           # these eval ids
#
# run_evals.py has no resume: evals that a dead pod interrupted start again in a new iteration-N (use FROM=).
set -euo pipefail
cd "$(dirname "$0")/.."                         # the pyterrier-skills repo

MODEL=${MODEL:-qwen-3.8-27b}
EFFORT=${EFFORT:-medium}
JOBS=${JOBS:-3}
PYTHON=${PYTHON:-python}
WORKSPACE=${WORKSPACE:-$(cd .. && pwd)/$(basename "$PWD")-evals}
IDS=${IDS:-}
FROM=${FROM:-}

if [ -n "$FROM" ] && [ -z "$IDS" ]; then        # the evals that did not finish in that iteration
  IDS=$("$PYTHON" - "$WORKSPACE/$FROM" <<'EOF'
import json, sys
from pathlib import Path
it = Path(sys.argv[1])
evals = json.load(open("evals/evals.json"))["evals"]
print(" ".join(str(e["id"]) for e in evals if not (it / f"eval-{e['id']}-{e['name']}" / "findings.md").exists()))
EOF
)
  [ -z "$IDS" ] && { echo "every eval in $FROM has findings.md; nothing to run"; exit 0; }
  echo "not finished in $FROM: $IDS"
fi

args=(--model "$MODEL" --effort "$EFFORT" --jobs "$JOBS" --workspace "$WORKSPACE")
for id in $IDS; do args+=(--id "$id"); done

mkdir -p logs
LOG=logs/evals-$(date +%Y%m%d-%H%M).log
echo "running run_evals.py ${IDS:+ids: $IDS}   log: $LOG"
"$PYTHON" -u evals/run_evals.py "${args[@]}" 2>&1 | tee "$LOG"
