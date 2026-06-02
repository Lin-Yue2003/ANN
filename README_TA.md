# TA Execution README

本文件給助教重現本專案的主要結果使用。內容包含環境建立、資料集放置、best config 執行方式、可調參數，以及完整 ablation 的執行指令。

## 1. 環境需求

建議使用 `uv` 管理 Python 環境。本專案已提供 `pyproject.toml` 和 `uv.lock`。

```bash
cd /path/to/term_project
uv sync
```

若不用 `uv`，也可以使用：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

需要的主要套件：

| 套件 | 用途 |
|---|---|
| `numpy` | 向量運算、exact rerank、argpartition |
| `h5py` | 讀取 SIFT HDF5 dataset |
| `hnswlib` | HNSW graph ANN index |
| `matplotlib` | 結果分析/繪圖輔助 |

## 2. Dataset

使用 ANN-benchmarks 的 SIFT-128 Euclidean HDF5 檔案。請將資料放在：

```text
data/sift-128-euclidean.hdf5
```

下載方式：

```bash
mkdir -p data
wget -P data https://storage.googleapis.com/ann-datasets/ann-benchmarks/sift-128-euclidean.hdf5
```

程式會讀取：

| Split | Shape | 說明 |
|---|---:|---|
| `train` | 1,000,000 x 128 | base vectors |
| `test` | 10,000 x 128 | query vectors |

Label 和 filter range 不是 dataset 原生欄位，而是用 seed 產生：

- base vector label: uniform integer in `[0, n_labels)`
- query filter: label range `[lo, hi]`
- filter selectivity: `(hi - lo + 1) / n_labels`

## 3. 快速 sanity check

先跑小資料確認環境、HNSW、CSV logging 都正常：

```bash
uv run -m main \
  --sift \
  --max-base 1000 \
  --n-query 50 \
  --method hnsw-filter-aug \
  --hnsw-ef-search 200 \
  --candidate-budget 300 \
  --hnsw-alpha 0.65 \
  --hnsw-label-dim-ratio 0.01 \
  --experiment-log experiments/ta_sanity.csv
```

成功時會印出：

- `Recall@50`
- `QPS`
- `Final Score`
- CSV log path

## 4. Best Config 重現

我們選用的 best config：

| Method | candidate_budget | ef_search | alpha | label_dim_ratio |
|---|---:|---:|---:|---:|
| `hnsw-filter-aug` | 300 | 200 | 0.65 | 0.01 |

完整 SIFT1M 重現指令：

```bash
uv run -m main \
  --sift \
  --max-base 1000000 \
  --n-query 10000 \
  --method hnsw-filter-aug \
  --k 50 \
  --n-labels 1000 \
  --min-sel 0.05 \
  --max-sel 0.40 \
  --seed 42 \
  --hnsw-m 16 \
  --hnsw-ef-construction 200 \
  --hnsw-ef-search 200 \
  --candidate-budget 300 \
  --hnsw-alpha 0.65 \
  --hnsw-label-dim-ratio 0.01 \
  --experiment-log experiments/best_config.csv
```

如果只想用預設讀完整 SIFT，也可以省略 `--max-base` 和 `--n-query`：

```bash
uv run -m main \
  --sift \
  --method hnsw-filter-aug \
  --seed 42 \
  --hnsw-ef-search 200 \
  --candidate-budget 300 \
  --hnsw-alpha 0.65 \
  --hnsw-label-dim-ratio 0.01 \
  --experiment-log experiments/best_config.csv
```

## 5. 助教可能會調整的變數

### Dataset / evaluation variables

| CLI | Default | 說明 |
|---|---:|---|
| `--seed` | `42` | 控制 labels、filter ranges、HNSW random seed |
| `--max-base` | `None` | base vector 數量；`None` 表示完整 SIFT1M |
| `--n-query` | `None` | query 數量；`None` 表示完整 10K |
| `--k` | `50` | Recall@K 的 K |
| `--n-labels` | `1000` | label 空間大小 |
| `--min-sel` | `0.05` | 最小 filter selectivity |
| `--max-sel` | `0.40` | 最大 filter selectivity |
| `--experiment-log` | `None` | CSV 輸出位置 |

### HNSW variables

| CLI | Default | 說明 |
|---|---:|---|
| `--hnsw-m` | `16` | HNSW graph 每個節點的連邊密度；較大通常 recall 較高、build/search 成本較高 |
| `--hnsw-ef-construction` | `200` | build graph 時的搜尋寬度；build time 不算進 QPS |
| `--hnsw-ef-search` | `200` | query 時的搜尋寬度；較大通常 recall 較高、QPS 較低 |
| `--candidate-budget` | `1000` | 每個 query 從 HNSW 取出的候選數，之後再 label filter + exact rerank |

### Filter-augmented HNSW variables

| CLI | Default | 說明 |
|---|---:|---|
| `--hnsw-alpha` | `0.6` | 向量距離權重；本 README 的 best config 使用 `0.65` |
| `--hnsw-label-dim-ratio` | `0.05` | 額外 label 維度數量比例；本 README 的 best config 使用 `0.01` |

### Adaptive variables

| CLI | Default | 說明 |
|---|---:|---|
| `--tau-small` | `0.01` | selectivity 小於等於此值時走 exact label-sorted search |
| `--tau-medium` | `0.15` | adaptive method 的 medium/large 門檻 |
| `--adaptive-medium-index` | `hnsw` | `adaptive-hnsw` 的 medium 分支，可選 `hnsw` 或 `lsh` |

### Follow-up ablation variables

| CLI | Default | 說明 |
|---|---:|---|
| `--adaptive-budget-tau-small` | `0.10` | `hnsw-filter-aug-budget` 的 small/medium budget 分界 |
| `--adaptive-budget-tau-medium` | `0.20` | `hnsw-filter-aug-budget` 的 medium/large budget 分界 |
| `--adaptive-budget-small` | `400` | small-selectivity query 的 HNSW candidates |
| `--adaptive-budget-medium` | `250` | medium-selectivity query 的 HNSW candidates |
| `--adaptive-budget-large` | `120` | large-selectivity query 的 HNSW candidates |
| `--hnsw-n-shards` | `20` | `hnsw-label-shards` 的 label shard 數量 |
| `--shard-min-budget` | `20` | 每個 shard 至少取出的 candidates |

## 6. 改 seed 取平均

如果助教想改 seed 並取平均，可以用同一個 CSV 累積多次結果。以下仍使用第 4 節的 best config：

```bash
for SEED in 40 41 42; do
  uv run -m main \
    --sift \
    --max-base 1000000 \
    --n-query 10000 \
    --method hnsw-filter-aug \
    --k 50 \
    --n-labels 1000 \
    --min-sel 0.05 \
    --max-sel 0.40 \
    --seed "${SEED}" \
    --hnsw-m 16 \
    --hnsw-ef-construction 200 \
    --hnsw-ef-search 200 \
    --candidate-budget 300 \
    --hnsw-alpha 0.65 \
    --hnsw-label-dim-ratio 0.01 \
    --experiment-log experiments/best_config_multiseed.csv
done
```

分析 CSV：

```bash
uv run -m analyze_results experiments/best_config_multiseed.csv
```

或直接用 Python 算平均：

```bash
uv run python - <<'PY'
import pandas as pd
df = pd.read_csv("experiments/best_config_multiseed.csv")
print(df[["method", "seed", "final_score", "mean_recall", "qps"]])
print()
print(df.groupby("method")[["final_score", "mean_recall", "qps"]].agg(["mean", "std"]))
PY
```

## 7. 完整 ablation v1

v1 會跑主要 HNSW ablation：

- `hnsw`
- `adaptive-hnsw`
- `hnsw-filter-aug`
- `adaptive-hnsw-aug`

主 sweep 已刻意排除 `hnsw-dynamic`，因為該方法會對同一個 query 重複呼叫 `hnswlib.knn_query` 來擴張 budget，QPS 明顯過低，不適合作為主要搜尋空間。

完整 SIFT1M：

```bash
RESULTS_CSV=experiments/hnsw_ablation_v1_full.csv \
LOG_FILE=experiments/hnsw_ablation_v1_full.log \
bash run_hnsw_ablation_uv.sh --sift --max-base 1000000 --n-query 10000
```

較小測試：

```bash
RESULTS_CSV=experiments/hnsw_ablation_v1_100k.csv \
LOG_FILE=experiments/hnsw_ablation_v1_100k.log \
bash run_hnsw_ablation_uv.sh --sift --max-base 100000 --n-query 1000
```

限制只跑前幾組 config：

```bash
uv run -m hnsw_ablation_runner \
  --sift \
  --max-base 100000 \
  --n-query 1000 \
  --limit-configs 10 \
  --experiment-log experiments/hnsw_ablation_v1_debug.csv
```

## 8. 完整 ablation v2

v2 是針對 v1 best region 的 focused search，並加入 multi-seed 驗證。已完成的 `hnsw_ablation_v2_full.csv` 包含 900 rows。

重用策略：

- SIFT vectors 只 load 一次
- 每個 seed 的 exact ground truth 只建一次
- 每個 `(seed, hnsw_alpha, hnsw_label_dim_ratio)` 只建一次 augmented HNSW graph
- 同一個 graph 會重複測不同 `hnsw_ef_search` 和 `candidate_budget`

預設 v2 搜尋：

| Parameter | Values |
|---|---|
| seeds | `40,41,42` |
| `hnsw_alpha` | `0.65, 0.75, 0.85, 0.90, 0.95` |
| `hnsw_label_dim_ratio` | `0.005, 0.01, 0.02` |
| `hnsw_ef_search` | `200, 300, 400, 500` |
| `candidate_budget` | `300, 400, 500, 600, 800` |

總共：

- 每個 seed 300 筆搜尋結果
- 每個 seed 15 次 graph build
- 三個 seed 共 900 筆搜尋結果、45 次 graph build

完整 SIFT1M：

```bash
RESULTS_CSV=experiments/hnsw_ablation_v2_full.csv \
LOG_FILE=experiments/hnsw_ablation_v2_full.log \
SEEDS=40,41,42 \
bash run_hnsw_ablation_v2_uv.sh --sift --max-base 1000000 --n-query 10000
```

較小測試：

```bash
RESULTS_CSV=experiments/hnsw_ablation_v2_100k.csv \
LOG_FILE=experiments/hnsw_ablation_v2_100k.log \
SEEDS=40,41,42 \
bash run_hnsw_ablation_v2_uv.sh --sift --max-base 100000 --n-query 1000
```

限制只跑前幾組 config：

```bash
uv run -m hnsw_ablation_v2_runner \
  --sift \
  --max-base 100000 \
  --n-query 1000 \
  --seeds 40,41,42 \
  --limit-configs 10 \
  --experiment-log experiments/hnsw_ablation_v2_debug.csv
```

## 9. Single-thread follow-up ablation

single-thread 後如果要測新的加速方向，可以跑這個獨立 sweep。它會保留原本 `hnsw-filter-aug` control，另外加入兩個 method：

- `hnsw-filter-aug-budget`: 依 filter selectivity 分成 small/medium/large，三組各用固定 candidate budget。
- `hnsw-label-shards`: 依 label range 建多個 HNSW shard，query 時只查和 filter range 重疊的 shard。

完整 SIFT1M：

```bash
RESULTS_CSV=experiments/hnsw_single_thread_followup_full.csv \
LOG_FILE=experiments/hnsw_single_thread_followup_full.log \
bash run_hnsw_single_thread_uv.sh --sift --max-base 1000000 --n-query 10000
```

較小測試：

```bash
RESULTS_CSV=experiments/hnsw_single_thread_followup_100k.csv \
LOG_FILE=experiments/hnsw_single_thread_followup_100k.log \
bash run_hnsw_single_thread_uv.sh --sift --max-base 100000 --n-query 1000
```

也可以只跑單一新 method：

```bash
uv run -m main \
  --sift \
  --max-base 100000 \
  --n-query 1000 \
  --method hnsw-filter-aug-budget \
  --hnsw-ef-search 200 \
  --candidate-budget 400 \
  --hnsw-alpha 0.65 \
  --hnsw-label-dim-ratio 0.01 \
  --adaptive-budget-small 400 \
  --adaptive-budget-medium 220 \
  --adaptive-budget-large 120 \
  --experiment-log experiments/hnsw_filter_aug_budget.csv
```

```bash
uv run -m main \
  --sift \
  --max-base 100000 \
  --n-query 1000 \
  --method hnsw-label-shards \
  --hnsw-ef-search 200 \
  --candidate-budget 300 \
  --hnsw-n-shards 20 \
  --shard-min-budget 20 \
  --experiment-log experiments/hnsw_label_shards.csv
```

## 10. CSV 欄位

每一筆實驗會 append 到 CSV。重要欄位：

| 欄位 | 說明 |
|---|---|
| `method` | 使用的方法 |
| `seed` | label/filter/HNSW seed |
| `n_base`, `n_query`, `k` | dataset/evaluation size |
| `n_labels`, `min_sel`, `max_sel`, `mean_sel` | filter 設定 |
| `hnsw_m`, `hnsw_ef_construction`, `hnsw_ef_search` | HNSW graph/query 參數 |
| `candidate_budget` | HNSW candidate pool size |
| `hnsw_alpha`, `hnsw_label_dim_ratio` | filter-augmented HNSW 參數 |
| `adaptive_budget_*` | `hnsw-filter-aug-budget` 的 selectivity bucket 參數 |
| `hnsw_n_shards`, `shard_min_budget` | `hnsw-label-shards` 參數 |
| `search_time_s` | 不含 index build 的 query time |
| `qps` | `n_query / search_time_s` |
| `mean_recall` | mean Recall@K against exact prefilter ground truth |
| `final_score` | `(QPS / 100) * mean_recall^2` |
| `avg_surviving_candidates` | label filter 後平均剩餘 candidates |
| `avg_survival_rate` | surviving / total candidates |
| `queries_with_zero_surviving` | label filter 後沒有 candidate 的 query 數 |

## 11. 注意事項

- QPS 不包含 index build time。這符合本專案評分設定，因此 HNSW build time 可以增加，但 query path 要快。
- 為了公平比較，HNSW index/query 固定使用單 thread；NumPy/BLAS 不額外限制 thread。
- `PreFilterSearch` 是 exact ground truth，不是競賽 method。
- `hnsw-filter-aug` 的 recall 可能低於 global HNSW，但 QPS 高很多，因此 final score 最高。
- 若助教要求較高 recall，可參考下一節的 best recall config。
- `hnsw-dynamic` 是失敗 ablation，保留在單次 CLI 中可測，但不建議放進主要 sweep。

## 12. Best Recall Config

如果想優先看 recall，可以跑下面這組。它的 score 會比第 4 節 best config 低，但 recall 較高。

| Method | candidate_budget | ef_search | alpha | label_dim_ratio |
|---|---:|---:|---:|---:|
| `hnsw-filter-aug` | 400 | 300 | 0.65 | 0.02 |

```bash
uv run -m main \
  --sift \
  --max-base 1000000 \
  --n-query 10000 \
  --method hnsw-filter-aug \
  --k 50 \
  --n-labels 1000 \
  --min-sel 0.05 \
  --max-sel 0.40 \
  --seed 42 \
  --hnsw-m 16 \
  --hnsw-ef-construction 200 \
  --hnsw-ef-search 300 \
  --candidate-budget 400 \
  --hnsw-alpha 0.65 \
  --hnsw-label-dim-ratio 0.02 \
  --experiment-log experiments/best_recall_config.csv
```
