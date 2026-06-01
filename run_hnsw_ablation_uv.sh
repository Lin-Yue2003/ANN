#!/usr/bin/env bash
set -euo pipefail

# HNSW ablation runner for uv environments.
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

RESULTS_CSV="${RESULTS_CSV:-experiments/hnsw_ablation_results.csv}"
LOG_FILE="${LOG_FILE:-experiments/hnsw_ablation_$(date +%Y%m%d_%H%M%S).log}"

if [ "$#" -gt 0 ]; then
  BASE_ARGS=("$@")
else
  BASE_ARGS=(--sift --max-base 100000 --n-query 1000)
fi

run_exp() {
  echo
  echo "================================================================"
  echo "RUN: uv run -m main ${BASE_ARGS[*]} $* --experiment-log ${RESULTS_CSV}"
  echo "================================================================"
  uv run -m main "${BASE_ARGS[@]}" "$@" --experiment-log "${RESULTS_CSV}"
}

{
  echo "[HNSW Ablation] started at $(date)"
  echo "[HNSW Ablation] results csv: ${RESULTS_CSV}"
  echo "[HNSW Ablation] log file: ${LOG_FILE}"
  echo "[HNSW Ablation] base args: ${BASE_ARGS[*]}"

  echo
  echo "### Phase 1: Method ablation"
  run_exp --method hnsw \
    --hnsw-ef-search 400 \
    --candidate-budget 1000

  run_exp --method adaptive-hnsw \
    --tau-small 0.01 \
    --tau-medium 0.15 \
    --hnsw-ef-search 400 \
    --candidate-budget 1000

  run_exp --method hnsw-dynamic \
    --hnsw-ef-search 400 \
    --initial-candidate-budget 200 \
    --candidate-budget 2000 \
    --budget-expansion-factor 2.0 \
    --min-survivors-multiplier 2.0

  run_exp --method hnsw-filter-aug \
    --hnsw-ef-search 400 \
    --candidate-budget 1000 \
    --hnsw-alpha 0.6 \
    --hnsw-label-dim-ratio 0.05

  run_exp --method adaptive-hnsw-aug \
    --tau-small 0.01 \
    --tau-medium 0.15 \
    --hnsw-ef-search 400 \
    --candidate-budget 1000 \
    --hnsw-alpha 0.6 \
    --hnsw-label-dim-ratio 0.05

  echo
  echo "### Phase 2A: Dynamic budget parameter search"
  for ef_search in 200 400 800; do
    for initial_budget in 100 200 500; do
      for max_budget in 1000 2000; do
        for min_survivors in 1.0 2.0 4.0; do
          run_exp --method hnsw-dynamic \
            --hnsw-ef-search "${ef_search}" \
            --initial-candidate-budget "${initial_budget}" \
            --candidate-budget "${max_budget}" \
            --budget-expansion-factor 2.0 \
            --min-survivors-multiplier "${min_survivors}"
        done
      done
    done
  done

  echo
  echo "### Phase 2B: Filter-augmented HNSW parameter search"
  for ef_search in 200 400 800; do
    for candidate_budget in 500 1000 2000; do
      for hnsw_alpha in 0.3 0.5 0.7 0.9; do
        for label_dim_ratio in 0.01 0.05 0.10; do
          run_exp --method hnsw-filter-aug \
            --hnsw-ef-search "${ef_search}" \
            --candidate-budget "${candidate_budget}" \
            --hnsw-alpha "${hnsw_alpha}" \
            --hnsw-label-dim-ratio "${label_dim_ratio}"
        done
      done
    done
  done

  echo
  echo "### Phase 2C: Adaptive augmented routing parameter search"
  for tau_small in 0.005 0.01 0.02; do
    for tau_medium in 0.10 0.15 0.20 0.30; do
      for candidate_budget in 1000 2000; do
        for hnsw_alpha in 0.5 0.7 0.9; do
          run_exp --method adaptive-hnsw-aug \
            --tau-small "${tau_small}" \
            --tau-medium "${tau_medium}" \
            --hnsw-ef-search 400 \
            --candidate-budget "${candidate_budget}" \
            --hnsw-alpha "${hnsw_alpha}" \
            --hnsw-label-dim-ratio 0.05
        done
      done
    done
  done

  echo
  echo "### Summary"
  uv run -m analyze_results "${RESULTS_CSV}"
  echo "[HNSW Ablation] finished at $(date)"
} 2>&1 | tee -a "${LOG_FILE}"
