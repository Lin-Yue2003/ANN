#!/bin/bash
# run_baseline.sh
# ===============
# Baseline test with default parameters and small dataset
# This run should complete in 1-2 minutes for quick validation

set -e

EXP_LOG="experiments/results.csv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "  Filtered ANNS - BASELINE TEST"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Output log: $EXP_LOG"
echo ""

# Create experiments directory if needed
mkdir -p experiments

# Small dataset baseline: help validate setup is working
echo "[1/3] Running baseline on SMALL dataset (validation)..."
python main.py \
    --max-base 10000 \
    --n-query 200 \
    --method postfilter \
    --alpha 0.05 \
    --label-dim-ratio 0.05 \
    --n-tables 400 \
    --n-functions 5 \
    --bin-width 0.22 \
    --experiment-log "$EXP_LOG"

echo ""
echo "[✓] Baseline test completed."
echo ""
echo "Next steps:"
echo "  1. Check results in: $EXP_LOG"
echo "  2. For small parameter sweep: bash run_small_sweep.sh"
echo "  3. For full parameter sweep: bash run_full_sweep.sh"
echo ""
