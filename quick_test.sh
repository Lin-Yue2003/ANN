#!/bin/bash
# quick_test.sh
# ==============
# Minimal test with synthetic data to verify framework is working
# Should complete in <30 seconds

set -e

echo "Quick Framework Test"
echo "==================="
echo ""

mkdir -p experiments

# Test 1: Import check
echo "[1/3] Checking imports..."
python -c "
from experiments.logger import ExperimentLogger
from adaptive_search import AdaptiveFilteredSearch
from label_bucket import LabelBucketIndex
print('✓ All modules import successfully')
"

# Test 2: Baseline command syntax
echo "[2/3] Testing parameter generation..."
CONFIG_COUNT=$(python sweep_params.py baseline | grep -c "^python main.py")
echo "✓ Generated $CONFIG_COUNT baseline configuration"

# Test 3: Tiny synthetic run
echo "[3/3] Running tiny test (N=1000, Q=50)..."
python main.py \
    --max-base 1000 \
    --n-query 50 \
    --k 50 \
    --method postfilter \
    --experiment-log experiments/test_results.csv \
    2>/dev/null

echo ""
echo "Test Results"
echo "============"
if [ -f experiments/test_results.csv ]; then
    LINES=$(wc -l < experiments/test_results.csv)
    echo "✓ CSV created with $LINES lines"
    
    # Show best score
    SCORE=$(tail -1 experiments/test_results.csv | cut -d, -f27)
    RECALL=$(tail -1 experiments/test_results.csv | cut -d, -f18)
    QPS=$(tail -1 experiments/test_results.csv | cut -d, -f20)
    
    echo ""
    echo "Latest result:"
    echo "  Recall:  $RECALL"
    echo "  QPS:     $QPS"
    echo "  Score:   $SCORE"
else
    echo "✗ CSV not created (check error above)"
    exit 1
fi

echo ""
echo "✓ Framework test PASSED"
echo ""
echo "Next steps:"
echo "  1. For remote execution: bash run_all.sh baseline"
echo "  2. For full sweep: bash run_all.sh full"
echo "  3. For results analysis: python analyze_results.py experiments/test_results.csv"
echo ""
