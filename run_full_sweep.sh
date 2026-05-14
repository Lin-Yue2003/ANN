#!/bin/bash
# run_full_sweep.sh
# =================
# Full parameter sweep on SIFT1M: ~500+ configurations
# Runtime: several hours depending on hardware
# 
# IMPORTANT: Requires SIFT1M dataset to be present in --sift-dir

set -e

EXP_LOG="experiments/results_full_sift.csv"
SIFT_ARGS="--sift --sift-dir ./data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="experiments/sweep_full_${TIMESTAMP}.log"

echo "=========================================="
echo "  Filtered ANNS - FULL PARAMETER SWEEP"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Output log: $EXP_LOG"
echo "Run log: $LOGFILE"
echo "Dataset: SIFT1M (full dataset)"
echo ""

mkdir -p experiments

# Check if SIFT data exists
if [ ! -d "./data" ] || [ -z "$(ls -A ./data 2>/dev/null)" ]; then
    echo "ERROR: SIFT data not found in ./data/"
    echo "Please download SIFT1M from: http://corpus-texmex.irisa.fr/"
    echo "Expected files:"
    echo "  ./data/sift_base.fvecs"
    echo "  ./data/sift_query.fvecs"
    exit 1
fi

echo "✓ SIFT data found"
echo ""

# Generate all commands
echo "Generating parameter configurations..."
COMMANDS=$(python sweep_params.py full $SIFT_ARGS)

CONFIG_COUNT=$(echo "$COMMANDS" | grep -c "^python main.py")
echo "Total configurations: $CONFIG_COUNT"
echo "Estimated runtime: $((CONFIG_COUNT * 30 / 60)) minutes (rough estimate)"
echo ""

# Execute each command
CONFIG_NUM=0
START_TIME=$(date +%s)
echo "Starting execution at $(date)..." | tee -a "$LOGFILE"

while IFS= read -r line; do
    if [[ $line == python* ]]; then
        ((CONFIG_NUM++))
        PERCENT=$((CONFIG_NUM * 100 / CONFIG_COUNT))
        ELAPSED=$(($(date +%s) - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        
        echo ""
        echo "[$PERCENT%] Config $CONFIG_NUM/$CONFIG_COUNT [${ELAPSED_MIN}m elapsed]"
        echo "$line"
        echo "---"
        
        # Execute command and append log
        eval "$line" --experiment-log "$EXP_LOG" 2>&1 | tee -a "$LOGFILE"
        
        # Print progress
        echo "[$(date +%H:%M:%S)] Completed: $CONFIG_NUM/$CONFIG_COUNT" | tee -a "$LOGFILE"
    fi
done <<< "$COMMANDS"

TOTAL_TIME=$(($(date +%s) - START_TIME))
TOTAL_MIN=$((TOTAL_TIME / 60))

echo ""
echo "=========================================="
echo "  FULL SWEEP COMPLETED"
echo "=========================================="
echo "Total runtime: ${TOTAL_MIN} minutes"
echo "Results saved to: $EXP_LOG"
echo "Run log saved to: $LOGFILE"
echo ""
echo "To analyze results:"
echo "  python analyze_results.py $EXP_LOG"
echo ""
