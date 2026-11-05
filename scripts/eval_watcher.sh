#!/usr/bin/env bash
# Evaluate each run on CPU as soon as its training finishes (history.json appears), while the GPU trains the next
# model. CPU is used deliberately: two concurrent MPS jobs exhausted the 8 GB unified memory.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUNS=("$@")
for run in "${RUNS[@]}"; do
  until [ -f "ml/artifacts/models/$run/history.json" ]; do sleep 30; done
  if [ -f "ml/artifacts/evaluation/$run/metrics.json" ]; then echo "EVAL_SKIP $run"; continue; fi
  echo "EVAL_START $run $(date +%H:%M)"
  if OMP_NUM_THREADS=3 MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python ml/evaluation/evaluate.py --run "$run" --device cpu --workers 2; then
    echo "EVAL_DONE $run $(date +%H:%M)"
  else
    echo "EVAL_FAILED $run"
  fi
done
echo "EVAL_ALL_DONE"
