#!/usr/bin/env bash
set -euo pipefail

# Focused v2 HNSW-filter-aug search.
# Reuses loaded vectors across all seeds, exact ground truth once per seed,
# and one augmented graph per (seed, hnsw_alpha, hnsw_label_dim_ratio).

mkdir -p experiments

RESULTS_CSV="${RESULTS_CSV:-experiments/hnsw_ablation_v2_results.csv}"
LOG_FILE="${LOG_FILE:-experiments/hnsw_ablation_v2_$(date +%Y%m%d_%H%M%S).log}"
SEEDS="${SEEDS:-40,41,42}"

if [ "$#" -gt 0 ]; then
  BASE_ARGS=("$@")
else
  BASE_ARGS=(--sift --max-base 100000 --n-query 1000)
fi

{
  echo "[HNSW Ablation V2] started at $(date)"
  echo "[HNSW Ablation V2] results csv: ${RESULTS_CSV}"
  echo "[HNSW Ablation V2] log file: ${LOG_FILE}"
  echo "[HNSW Ablation V2] seeds: ${SEEDS}"
  echo "[HNSW Ablation V2] base args: ${BASE_ARGS[*]}"
  echo
  echo "RUN: uv run -m hnsw_ablation_v2_runner ${BASE_ARGS[*]} --seeds ${SEEDS} --experiment-log ${RESULTS_CSV}"

  uv run -m hnsw_ablation_v2_runner \
    "${BASE_ARGS[@]}" \
    --seeds "${SEEDS}" \
    --experiment-log "${RESULTS_CSV}"

  echo
  echo "### Summary"
  uv run -m analyze_results "${RESULTS_CSV}"
  echo "[HNSW Ablation V2] finished at $(date)"
} 2>&1 | tee -a "${LOG_FILE}"
