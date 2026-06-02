#!/usr/bin/env bash
set -euo pipefail

RESULTS_CSV="${RESULTS_CSV:-experiments/hnsw_single_thread_followup.csv}"
LOG_FILE="${LOG_FILE:-experiments/hnsw_single_thread_followup.log}"

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${LOG_FILE}")"

echo "[run] Writing CSV to ${RESULTS_CSV}"
echo "[run] Writing log to ${LOG_FILE}"

uv run -m hnsw_ablation_runner \
  --sweep hnsw-single-thread \
  --experiment-log "${RESULTS_CSV}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
