# Filtered ANNS Optimization Framework

## Quick Start

### 1. Baseline Validation (1-2 minutes)
```bash
bash run_all.sh baseline
```
This runs a quick test with default parameters on a small dataset (10K base, 200 queries).

### 2. Small Parameter Sweep (20-40 minutes)
```bash
bash run_all.sh small
```
This runs ~50 configurations on a medium dataset (50K base, 500 queries).

### 3. Full SIFT Sweep (several hours)
```bash
bash run_all.sh full
```
This runs ~500+ configurations on the full SIFT1M dataset.

### 4. Progressive Execution (All layers)
```bash
bash run_all.sh progressive
```
Runs baseline → small → full sequentially.

---

## Detailed Usage

### Step-by-step for Remote Execution

When running on a remote machine:

#### 1. Initial Setup
```bash
# Transfer project to remote
scp -r . remote-machine:/path/to/project/

# On remote machine
cd /path/to/project/
chmod +x run_all.sh run_baseline.sh run_small_sweep.sh run_full_sweep.sh

# Optional: for SIFT experiments, ensure data is present
# Download from: http://corpus-texmex.irisa.fr/
# Should have: data/sift_base.fvecs, data/sift_query.fvecs
```

#### 2. Choose Execution Layer

**Option A: Baseline only** (validation that setup works)
```bash
bash run_all.sh baseline
```

**Option B: Small sweep** (representative subset)
```bash
bash run_all.sh small
```

**Option C: Full sweep** (comprehensive optimization)
```bash
bash run_all.sh full
```

#### 3. Monitor Progress
The scripts log to:
- `experiments/results.csv` - Main results file (append mode)
- `experiments/sweep_TIMESTAMP.log` - Detailed execution log

While running:
```bash
tail -f experiments/results.csv
tail -f experiments/sweep_*.log
```

#### 4. Download Results
```bash
# On local machine
scp -r remote-machine:/path/to/project/experiments/ ./

# Analyze
python analyze_results.py experiments/results.csv
```

---

## Results Analysis

### Automatic Analysis
```bash
python analyze_results.py experiments/results.csv
```

This prints:
- Experiments count by method
- Top 3 configurations per method
- Overall best configuration
- Reproduction command

### Manual Inspection
```bash
# View raw CSV
head -20 experiments/results.csv

# Extract specific columns
cut -d, -f1,2,14,15,16,17 experiments/results.csv | column -t -s,

# Sort by final_score
tail -n +2 experiments/results.csv | sort -t, -k32 -nr | head -5
```

---

## Parameter Configuration

### Methods Available
- `postfilter`: Standard filter-augmented LSH
- `adaptive`: Selectivity-based hybrid routing
- `label-sorted`: Exact pre-filter with label bucket acceleration

### Default Parameters

#### Filter-augmented LSH (postfilter)
```
--alpha 0.05                    # Filter-augmentation weight
--label-dim-ratio 0.05          # Label vector dimension ratio
--n-tables 400                  # Number of LSH tables
--n-functions 5                 # Hash functions per table
--bin-width 0.22                # LSH bin width
```

#### Adaptive Search
```
--tau-small 0.01                # Selectivity threshold for exact search
--tau-medium 0.15               # Selectivity threshold for LSH
```

#### Multi-probe (placeholder)
```
--probe-radius 0                # 0=disabled, 1=±1 per dimension
--max-extra-probes 0
```

### Custom Experiment

Run a single custom configuration:
```bash
python main.py \
  --max-base 50000 \
  --n-query 500 \
  --method adaptive \
  --alpha 0.1 \
  --n-tables 300 \
  --bin-width 0.25 \
  --tau-small 0.01 \
  --tau-medium 0.20 \
  --experiment-log experiments/custom.csv
```

---

## Output Files

After running:

### Main Results
- `experiments/results.csv` - Append-only results (all runs combined)
- `experiments/results_full_sift.csv` - Full SIFT results (created by `run_full_sweep.sh`)

### Logs
- `experiments/sweep_TIMESTAMP.log` - Detailed execution logs (one per run)

### Generated Commands (temporary)
```bash
python sweep_params.py baseline    # Show baseline commands
python sweep_params.py small       # Show small sweep commands
python sweep_params.py full        # Show full sweep commands
```

---

## Troubleshooting

### Script Permission Denied
```bash
chmod +x run_all.sh run_baseline.sh run_small_sweep.sh run_full_sweep.sh
```

### SIFT Data Not Found
```bash
# For full SIFT experiments, data must be in ./data/
# Download from: http://corpus-texmex.irisa.fr/
# After download:
mkdir -p data
gunzip sift_base.fvecs.gz
gunzip sift_query.fvecs.gz
# Now run: bash run_all.sh full
```

### Out of Memory
Use smaller dataset:
```bash
python main.py \
  --max-base 100000 \
  --n-query 100 \
  --method adaptive
```

### Slow Execution
- Reduce number of configurations: edit `sweep_params.py`
- Use smaller dataset: edit `run_small_sweep.sh`
- Run only specific method: pass `--method` flag

---

## Framework Architecture

```
run_all.sh (master orchestrator)
  ├── run_baseline.sh
  ├── run_small_sweep.sh
  └── run_full_sweep.sh

sweep_params.py (config generator)
  └── Outputs commands for each configuration

main.py (single experiment runner)
  ├── Loads data
  ├── Runs method (postfilter/adaptive/label-sorted)
  ├── Logs results to CSV
  └── Prints detailed comparison

analyze_results.py (results analyzer)
  └── Summarizes best configurations
```

---

## Expected Runtimes

| Layer | Configs | Dataset | Est. Time |
|-------|---------|---------|-----------|
| Baseline | 1 | 10K base, 200Q | 1-2 min |
| Small | ~50 | 50K base, 500Q | 20-40 min |
| Full | ~550 | SIFT1M full | 8-16 hours |

---

## Citing Results

When reporting optimization results, include:
```
Method: [postfilter | adaptive | label-sorted]
Parameters: [all hyperparameters]
Final Score: S = (QPS/100) * R^2
  - Recall@50: R
  - QPS: Q
Timestamp: [from CSV header]
Dataset: [sift | synthetic]
```
