#!/bin/bash
# run_baseline.sh
# ===============
# Baseline test with default parameters and small dataset
# This run should complete in 1-2 minutes for quick validation

set -e

EXP_LOG="experiments/results.csv"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="experiments/run_baseline_${TIMESTAMP}.log"

echo "=========================================="
echo "  Filtered ANNS - BASELINE TEST"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Output log: $EXP_LOG"
echo "Run log: $LOGFILE"
echo ""

# 確保實驗結果資料夾存在
mkdir -p experiments

echo "[1/3] Running baseline on SMALL dataset (validation)..."
# 注意：這裡已經將參數改回符合 main.py argparse 定義的 --lsh-* 格式
python main.py \
    --max-base 10000 \
    --n-query 200 \
    --method postfilter \
    --alpha 0.05 \
    --label-dim-ratio 0.05 \
    --lsh-tables 400 \
    --lsh-functions 5 \
    --lsh-bin-width 0.22 \
    --experiment-log "$EXP_LOG" 2>&1 | tee -a "$LOGFILE"

# 檢查 Python 指令的執行狀態
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: Baseline test failed. Please check $LOGFILE for details."
    exit 1
fi

echo ""
echo "[✓] Baseline test completed."
echo ""
echo "Next steps:"
echo "  1. Check results in: $EXP_LOG"
echo "  2. For small parameter sweep: bash run_small_sweep.sh"
echo "  3. For full parameter sweep: bash run_full_sweep.sh"
echo ""