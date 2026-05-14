#!/bin/bash
# run_all.sh
# ==========
# Master orchestration script for all experiment layers
# 
# Usage:
#   ./run_all.sh baseline         # Run quick baseline validation
#   ./run_all.sh small            # Run small parameter sweep (medium scale)
#   ./run_all.sh full             # Run full SIFT sweep (largest scale)
#   ./run_all.sh progressive      # Run baseline → small → full (sequentially)

set -e

# 1. 先確保 experiments 目錄存在，避免接下來的 log 寫入失敗
mkdir -p experiments

# 2. 全域錯誤捕捉：將所有的 stdout 與 stderr 同時顯示在螢幕，並寫入 run_all_global.log
exec > >(tee -i experiments/run_all_global.log)
exec 2>&1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default layer
LAYER="${1:-baseline}"

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  Filtered ANNS Experiment Framework   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Layer: $LAYER"
echo "Date:  $(date)"
echo ""

case "$LAYER" in
    baseline)
        echo "==> Running: BASELINE (quick validation)"
        bash run_baseline.sh
        ;;
    
    small)
        echo "==> Running: SMALL SWEEP (medium-scale, ~50 configs)"
        bash run_small_sweep.sh
        ;;
    
    full)
        echo "==> Running: FULL SWEEP (SIFT1M, ~500+ configs)"
        bash run_full_sweep.sh
        ;;
    
    progressive)
        echo "==> Running: PROGRESSIVE (all layers sequentially)"
        echo ""
        echo "Step 1/3: Baseline..."
        bash run_baseline.sh
        
        echo ""
        echo "Step 2/3: Small sweep..."
        bash run_small_sweep.sh
        
        echo ""
        echo "Step 3/3: Full sweep..."
        bash run_full_sweep.sh
        
        echo ""
        echo "✓ ALL LAYERS COMPLETED"
        ;;
    
    *)
        echo "ERROR: Unknown layer: $LAYER"
        echo ""
        echo "Usage:"
        echo "  ./run_all.sh baseline    - Quick validation (1-2 min)"
        echo "  ./run_all.sh small       - Medium sweep (20-40 min)"
        echo "  ./run_all.sh full        - Full SIFT sweep (several hours)"
        echo "  ./run_all.sh progressive - All layers sequentially"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  Framework Execution Summary          ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Results files:"
ls -lh experiments/*.csv 2>/dev/null || echo "  (No results yet)"
echo ""
echo "To download results:"
echo "  scp -r experiments/ <your-machine>:"
echo ""