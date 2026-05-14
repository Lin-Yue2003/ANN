#!/bin/bash
# run_small_sweep.sh
# ==================
# Small parameter sweep: ~50 configurations for medium-scale validation
# Runtime: ~20-40 minutes depending on hardware

set -e

EXP_LOG="experiments/results.csv"
DATASET_SIZE="--max-base 50000 --n-query 500"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="experiments/sweep_small_${TIMESTAMP}.log"

echo "=========================================="
echo "  Filtered ANNS - SMALL PARAMETER SWEEP"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Output log: $EXP_LOG"
echo "Run log: $LOGFILE"
echo "Dataset: medium (50K base, 500 queries)"
echo ""

mkdir -p experiments

# Generate all commands
echo "Generating parameter configurations..."
COMMANDS=$(python sweep_params.py small $DATASET_SIZE)

CONFIG_COUNT=$(echo "$COMMANDS" | grep -c "^python main.py")
echo "Total configurations: $CONFIG_COUNT"
echo ""

# Execute each command
CONFIG_NUM=0
echo "Starting execution..." | tee -a "$LOGFILE"

while IFS= read -r line; do
    if [[ $line == python* ]]; then
        ((CONFIG_NUM++))
        PERCENT=$((CONFIG_NUM * 100 / CONFIG_COUNT))
        echo ""
        echo "[$PERCENT%] Config $CONFIG_NUM/$CONFIG_COUNT"
        echo "$line"
        echo "---"
        
        # Execute command and append log
        eval "$line" --experiment-log "$EXP_LOG" 2>&1 | tee -a "$LOGFILE"
    fi
done <<< "$COMMANDS"

echo ""
echo "=========================================="
echo "  SWEEP COMPLETED"
echo "=========================================="
echo "Results saved to: $EXP_LOG"
echo "Run log saved to: $LOGFILE"
echo ""
echo "To analyze results:"
echo "  python analyze_results.py $EXP_LOG"
echo ""
