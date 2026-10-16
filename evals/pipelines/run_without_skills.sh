#!/usr/bin/env bash
# The four pipeline-matrix conditions without skills, each as its own background process and log,
# added to the batch that already holds the `all` smoke run, so status/grade/report compare them.
#
#   bash evals/pipelines/run_without_skills.sh          # start all four (one line each, below)
#   bash evals/pipelines/run_without_skills.sh docs     # start only one condition
#
# Default: the smoke run's cells (BM25 and BM25>>RM3>>BM25 on NanoBEIR), CPU only; a cpu-lg node is enough.
# Settings can be overridden from the environment, e.g. the full matrix on a GPU node in a new batch:
#   NODE=a6000-xxl BATCH=pipelines-2 PIPELINES= DATASETS= bash evals/pipelines/run_without_skills.sh docs
# NODE is the cluster's size name; it goes into the prompt next to the detected GPU, CPU, RAM and disk.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # the pyterrier-skills repo

MODEL=${MODEL:-qwen-3.8-27b}
BATCH=${BATCH:-pipelines-1}                     # the batch with the finished `all` smoke run
PIPELINES=${PIPELINES-bm25,bm25_prf_bm25}       # empty = all 9 pipelines
DATASETS=${DATASETS-nanobeir}                   # empty = all datasets
NODE=${NODE:-}
mkdir -p logs/pipelines

run() {   # run <condition>
  nohup python evals/run_pipelines.py --model "$MODEL" --effort medium --iteration "$BATCH" \
    --conditions "$1" ${PIPELINES:+--pipelines "$PIPELINES"} ${DATASETS:+--datasets "$DATASETS"} ${NODE:+--node "$NODE"} \
    > "logs/pipelines/$BATCH-$1.log" 2>&1 &
  echo "started $1 (pid $!) -> logs/pipelines/$BATCH-$1.log"
  sleep 5                                       # let each process register in run_meta.json
}

if [ $# -gt 0 ]; then run "$1"; exit 0; fi

run none        # no skills, no documentation, no code view
run code        # code view only
run docs        # documentation only
run docs_code   # documentation + code view
# run skills       # skills only           (the remaining skills conditions, for the full comparison)
# run skills_code  # skills + code view
# run skills_docs  # skills + documentation
