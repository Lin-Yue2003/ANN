#!/usr/bin/env bash
set -euo pipefail

# HNSW ablation runner for uv environments.
# This uses hnsw_ablation_runner.py so data loading, label generation,
# filter range generation, and exact ground truth are done once per run.
# The main sweep intentionally excludes hnsw-dynamic because that method
# expands candidate budgets by repeating hnswlib queries for the same query.
#
# Default dataset:
#   --sift --max-base 100000 --n-query 1000
#
# Override by passing base args after the script name, for example:
#   bash run_hnsw_ablation_uv.sh --sift
#
# Override output paths with env vars:
#   RESULTS_CSV=experiments/my_results.csv LOG_FILE=experiments/my_run.log bash run_hnsw_ablation_uv.sh

mkdir -p experiments

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RESULTS_CSV="${RESULTS_CSV:-experiments/hnsw_ablation_results.csv}"
LOG_FILE="${LOG_FILE:-experiments/hnsw_ablation_$(date +%Y%m%d_%H%M%S).log}"

if [ "$#" -gt 0 ]; then
  BASE_ARGS=("$@")
else
  BASE_ARGS=(--sift --max-base 100000 --n-query 1000)
fi

{
  echo "[HNSW Ablation] started at $(date)"
  echo "[HNSW Ablation] results csv: ${RESULTS_CSV}"
  echo "[HNSW Ablation] log file: ${LOG_FILE}"
  echo "[HNSW Ablation] base args: ${BASE_ARGS[*]}"

  echo
  echo "RUN: uv run -m hnsw_ablation_runner ${BASE_ARGS[*]} --experiment-log ${RESULTS_CSV}"
  uv run -m hnsw_ablation_runner "${BASE_ARGS[@]}" --experiment-log "${RESULTS_CSV}"

  echo
  echo "### Summary"
  uv run -m analyze_results "${RESULTS_CSV}"
  echo "[HNSW Ablation] finished at $(date)"
} 2>&1 | tee -a "${LOG_FILE}"
