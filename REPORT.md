# Filtered ANNS with HNSW Graph Index: Experiment Report

## 摘要

本專案研究 filtered approximate nearest neighbor search。任務是對每個 query vector 找出符合 label range `[lo, hi]` 的 top-K nearest neighbors，並在速度與 recall 之間取得最佳平衡。

評分公式固定為：

```text
Final Score S = (QPS / 100) * Recall@K^2
```

其中 QPS 不包含 index build time。因此本專案的核心策略是：在 index build 階段建立更強的 graph index，讓 query 階段用較小 candidate pool 達到足夠 recall，並盡可能提高 QPS。

目前最好的 full SIFT1M 結果來自 v2 focused multi-seed search。v2 的 runner 在 CSV 中記為 `hnsw-filter-aug-v2`，代表同一個 `hnsw-filter-aug` 方法搭配 graph reuse runner 進行多 seed 搜尋；單次重現時使用 `main.py --method hnsw-filter-aug` 即可。

| Rank | Method | Final Score | Recall@50 | QPS | Key Config |
|---:|---|---:|---:|---:|---|
| 1 | `hnsw-filter-aug-v2` mean best | 115.5122 | 0.9091 | 13978.43 | `budget=300`, `ef=200`, `alpha=0.65`, `label_dim_ratio=0.01` |
| 2 | `hnsw-filter-aug-v2` single best | 118.0396 | 0.9101 | 14251.00 | seed 40, `budget=300`, `ef=200`, `alpha=0.65`, `label_dim_ratio=0.005` |
| 3 | v1 `hnsw-filter-aug` best | 95.5780 | 0.9267 | 11128.45 | seed 42, `budget=500`, `ef=400`, `alpha=0.9`, `label_dim_ratio=0.01` |
| 4 | v1 `hnsw` best | 63.6829 | 0.9703 | 6764.14 | `budget=1000`, `ef=400` |
| 5 | v1 `adaptive-hnsw-aug` best | 58.8848 | 0.9981 | 5911.19 | high recall, lower QPS |

相對於 v1 global `hnsw` baseline，v2 multi-seed mean best：

- Final score 提升約 `+81.4%`
- QPS 提升約 `+106.7%`
- Recall@50 下降約 `-0.0612` absolute

相對於 v1 `hnsw-filter-aug` best，v2 multi-seed mean best：

- Final score 提升約 `+20.9%`
- QPS 提升約 `+25.6%`
- Recall@50 下降約 `-0.0177` absolute

這代表在目前 score 公式下，把 candidate budget 進一步降到 300，並用更平衡的 `alpha=0.65` 讓 graph 同時保留 vector 與 label 訊號，比 v1 的 `budget=500`, `alpha=0.9` 更有效。

## 1. 問題定義

標準 ANN search 給定 query vector `q`，在 base vectors `X = {x_i}` 中找出 L2 distance 最近的 K 個點。Filtered ANN 額外要求每個 base vector 有 label `a_i`，只允許：

```text
lo <= a_i <= hi
```

因此每個 query 的 ground truth 是：

```text
topK_by_L2({x_i | lo <= a_i <= hi}, q)
```

本專案使用 exact prefilter 作為 ground truth，再用不同 ANN 方法產生 candidates，最後計算 Recall@K。

## 2. Dataset 與實驗設定

Full run 使用 ANN-benchmarks SIFT-128 Euclidean HDF5：

| Item | Value |
|---|---:|
| base vectors | 1,000,000 |
| query vectors | 10,000 |
| dimension | 128 |
| K | 50 |
| distance | squared L2 |
| labels | synthetic uniform integer labels |
| `n_labels` | 1000 |
| filter selectivity | 5% 至 40% |
| default seed | 42 |

Dataset 檔案：

```text
data/sift-128-euclidean.hdf5
```

Label 與 filter range 生成：

- `assign_labels(N, n_labels=1000, seed=seed)`
- `generate_filter_ranges(Q, min_selectivity=0.05, max_selectivity=0.40, seed=seed+1)`

也就是說，改變 `--seed` 會同時改變：

- base vector labels
- query filter ranges
- HNSW construction seed

## 3. 評估指標

### Recall@K

對每個 query：

```text
Recall@K = |ANN_result intersect Exact_result| / |Exact_result|
```

最後取所有 queries 的平均，記為 `mean_recall`。

### QPS

```text
QPS = n_query / search_time_s
```

注意：index build time 不算進 QPS。這是本專案選擇 HNSW graph index 的主要原因之一，因為 HNSW 可以把成本放在 build 階段，換取更快的 query。

### Final Score

```text
Final Score = (QPS / 100) * mean_recall^2
```

Recall 以平方進入公式，因此 recall 下降會被懲罰；但若 QPS 提升足夠大，仍可能得到更高 final score。

## 4. 系統架構

主要檔案：

| File | 功能 |
|---|---|
| `main.py` | 單次實驗入口，支援所有 method 與 CSV logging |
| `prefilter.py` | exact prefilter ground truth |
| `postfilter.py` | LSH postfilter baseline |
| `lsh_index.py` | E2LSH implementation |
| `hnsw_index.py` | `hnswlib` thin wrapper |
| `hnsw_search.py` | HNSW based filtered search methods |
| `hnsw_ablation_runner.py` | v1 ablation runner，重用 data load 與 exact ground truth |
| `hnsw_ablation_v2_runner.py` | v2 focused multi-seed runner，額外重用 graph |
| `sweep_params.py` | sweep config generator |
| `experiments/logger.py` | CSV logger |
| `analyze_results.py` | CSV result summary |

## 5. 方法設計

### 5.1 Exact PreFilter

方法：

```text
filter labels first -> exact L2 topK on surviving vectors
```

優點：

- exact recall = 1.0
- 可作為 ground truth
- filter 很小時候 candidates 少，速度可接受

缺點：

- filter 較寬時，每個 query 要掃大量 base vectors
- full SIFT1M 上不適合作為高 QPS search method

### 5.2 LabelSorted Exact Search

檔案：`hnsw_search.py` 的 `LabelSortedPreFilterSearch`

做法：

1. index build 時依 label 排序 base ids。
2. 建立每個 label 的 start/end boundary。
3. query 時用 `[lo, hi]` 直接切出一段 ids。
4. 對這段 ids 做 exact L2 rerank。

目的：

- 保留 exact semantics。
- 避免每個 query 都對全部 labels 做 boolean mask。
- 在 adaptive method 的 small-selectivity branch 中使用。

### 5.3 LSH PostFilter

方法：

```text
global LSH candidates -> label filter -> exact L2 rerank
```

LSH 是 E2LSH，適合 L2 distance。主要參數：

- `--lsh-tables`
- `--lsh-functions`
- `--lsh-bin-width`

優點：

- query 快。
- 作為原始 ANN baseline 清楚。

缺點：

- filter 較嚴格時，LSH 找到的 candidates 可能大多被 label filter 丟掉。
- candidate survival rate 低時 recall 會明顯下降。

### 5.4 Global HNSW PostFilter

檔案：

- `hnsw_index.py`
- `hnsw_search.py` 的 `HNSWPostFilterSearch`

方法：

```text
global HNSW candidates -> label filter -> exact L2 rerank
```

HNSW build：

```python
hnswlib.Index(space="l2", dim=D)
index.init_index(max_elements=N, ef_construction=..., M=..., random_seed=...)
index.add_items(base_vecs, ids)
index.set_ef(max(ef_search, candidate_budget))
```

Query：

1. `hnsw.knn_query(query, k=candidate_budget)`
2. 保留 label in `[lo, hi]` 的 candidate ids
3. 對 surviving candidates 用 NumPy exact squared L2 rerank
4. 回傳 top-K ids

為什麼保留 exact rerank：

- HNSW 只負責 candidate generation。
- 最後排序仍用原始 vector 的 exact L2。
- 不改變 metric 定義，也避免 augmented space 直接影響最終排名。

### 5.5 HNSW Dynamic Budget

方法：

```text
start small candidate budget
if surviving candidates too few:
    query HNSW again with larger budget
repeat until enough survivors or max budget
```

原本假設：

- filter 很寬時用小 budget 可以省時間。
- filter 很窄時才擴張 budget。

實驗結果顯示這個方向失敗。原因是 `hnswlib.knn_query` 每次呼叫都有固定成本，dynamic method 對同一個 query 重複查 HNSW，導致 QPS 嚴重下降。v1 後續主 sweep 已排除此方法，但仍保留作為失敗 ablation。

### 5.6 Filter-Augmented HNSW

檔案：`hnsw_search.py` 的 `HNSWFilterAugmentedSearch`

核心想法：把 label 資訊放入 HNSW graph 的建圖空間，使 graph traversal 更容易走到 label range 附近的點。最後仍回到原始 vector 做 exact rerank。

Base vector augmentation：

```text
x_aug = [sqrt(alpha) * x,
         sqrt(1 - alpha) * repeat(label / n_labels, label_dim)]
```

Query augmentation：

```text
mid = (lo + hi) / 2
q_aug = [sqrt(alpha) * q,
         sqrt(1 - alpha) * repeat(mid / n_labels, label_dim)]
```

其中：

```text
label_dim = max(1, round(D * hnsw_label_dim_ratio))
```

參數意義：

| Parameter | 意義 |
|---|---|
| `hnsw_alpha` | 原始 vector distance 權重 |
| `hnsw_label_dim_ratio` | 額外 label 維度比例 |

設計理由：

- 若只用 global HNSW，candidate label 分布接近全域 label 分布，filter 後可能浪費 candidates。
- 若把 label 直接放進 index space，HNSW candidate generation 會偏向 query filter 的 midpoint。
- 由於最後會做 label filter 和原始 L2 rerank，augmentation 只影響 candidate pool，不改變最終 metric。

v1 初步顯示 `alpha=0.9`, `label_dim_ratio=0.01` 很強；v2 細搜後發現 multi-seed 平均最佳轉移到 `alpha=0.65`, `label_dim_ratio=0.01`，而單次最高分出現在 `alpha=0.65`, `label_dim_ratio=0.005`。這代表 label signal 需要存在，但不能用過大的 label dimension；同時 `alpha` 不宜過高，否則 graph 太接近純 vector HNSW，無法充分利用 filter 資訊。

### 5.7 Adaptive HNSW

檔案：`hnsw_search.py` 的 `AdaptiveHNSWSearch`

Routing：

```text
if selectivity <= tau_small:
    exact label-sorted search
elif selectivity <= tau_medium:
    HNSW or LSH medium branch
else:
    global HNSW
```

目的：

- 小 filter：exact scan candidates 少，recall 最高。
- 中大 filter：用 ANN 提高 QPS。

結果：

- Recall 比 pure HNSW 稍高。
- 但 exact branch 和 routing overhead 使 QPS 較低，final score 沒有超過 `hnsw-filter-aug`。

### 5.8 Adaptive HNSW Augmented

檔案：`hnsw_search.py` 的 `AdaptiveHNSWAugmentedSearch`

Routing：

```text
small filters -> exact label-sorted search
medium filters -> filter-augmented HNSW
large filters -> global HNSW
```

目的：

- 把 `hnsw-filter-aug` 的 label-aware candidate generation 用在最需要 filter awareness 的 medium selectivity。
- 大 filter 則回到 global HNSW，避免 label augmentation 過度限制 vector neighborhood。

結果：

- 最高 recall 約 0.998。
- 但要同時建 global HNSW 和 augmented HNSW，且 query routing 較複雜，QPS 較低。
- 若評分更重視 recall，這是可考慮的方法；以目前 score 公式則不是最佳。

## 6. 實作最佳化

### 6.1 Batch Query

HNSW query 盡量使用 batch：

```python
labels, distances = index.knn_query(query_vecs, k=k)
```

避免 Python loop 對每個 query 呼叫 HNSW。

### 6.2 Exact Rerank

Candidate rerank 使用：

```python
diff = base_vecs[valid] - query
dists = np.einsum("nd,nd->n", diff, diff)
top_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
top_idx = top_idx[np.argsort(dists[top_idx])]
```

理由：

- `argpartition` 比完整 sort 快。
- 只對 candidate pool rerank，不掃全資料。
- 最終排名仍是 exact L2。

### 6.3 Graph Reuse in v2

v2 新增 `HNSWIndex.set_ef_search()`，允許在不重建 graph 的情況下改變 query-time `ef_search`。

v2 runner 的重用單位：

```text
(seed, hnsw_alpha, hnsw_label_dim_ratio)
```

同一個 graph 可測：

- 不同 `hnsw_ef_search`
- 不同 `candidate_budget`

這大幅減少 focused search 的重複 graph build。

## 7. Ablation v1 設計

v1 full ablation 使用：

```text
dataset = SIFT1M
n_query = 10000
k = 50
n_labels = 1000
min_sel = 0.05
max_sel = 0.40
seed = 42
```

CSV：`full_hnsw_ablation.csv`

總實驗數：200 rows

| Method | Runs |
|---|---:|
| `hnsw-filter-aug` | 110 |
| `adaptive-hnsw-aug` | 74 |
| `hnsw-dynamic` | 12 |
| `hnsw` | 2 |
| `adaptive-hnsw` | 2 |

注意：後續主 runner 已移除 `hnsw-dynamic`，因為 v1 顯示它的 QPS 過低。

## 8. Ablation v1 結果

### 8.1 Method-level Best Ranking

| Rank | Method | Best Score | Recall@50 | QPS | Candidate Budget | ef_search | Extra Config |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `hnsw-filter-aug` | 95.5780 | 0.9267 | 11128.45 | 500 | 400 | `alpha=0.9`, `label_dim_ratio=0.01` |
| 2 | `hnsw` | 63.6829 | 0.9703 | 6764.14 | 1000 | 400 | global postfilter |
| 3 | `adaptive-hnsw` | 58.9223 | 0.9806 | 6128.00 | 1000 | 400 | `tau_small=0.01`, `tau_medium=0.15` |
| 4 | `adaptive-hnsw-aug` | 58.8848 | 0.9981 | 5911.19 | 1000 | 400 | `alpha=0.5`, `label_dim_ratio=0.05` |
| 5 | `hnsw-dynamic` | 2.9466 | 0.9703 | 313.00 | 1000 | 200 | repeated query expansion |

### 8.2 Method-level Aggregate

| Method | Runs | Max Score | Mean Score | Median Score | Mean Recall | Mean QPS |
|---|---:|---:|---:|---:|---:|---:|
| `hnsw-filter-aug` | 110 | 95.5780 | 46.8496 | 43.4301 | 0.9848 | 4875.64 |
| `hnsw` | 2 | 63.6829 | 59.3049 | 59.3049 | 0.9703 | 6299.25 |
| `adaptive-hnsw` | 2 | 58.9223 | 56.7330 | 56.7330 | 0.9806 | 5900.38 |
| `adaptive-hnsw-aug` | 74 | 58.8848 | 38.9858 | 35.7822 | 0.9986 | 3912.61 |
| `hnsw-dynamic` | 12 | 2.9466 | 1.5505 | 1.5148 | 0.9781 | 162.86 |

解讀：

- `hnsw-filter-aug` 的 best score 遠高於其他方法，但平均 score 不代表方法弱，而是 sweep 包含很多不佳參數。
- `adaptive-hnsw-aug` recall 平均最高，但 QPS 明顯低，因此 score 低於 `hnsw-filter-aug`。
- `hnsw-dynamic` recall 尚可，但 QPS 崩潰，確認 repeated HNSW query 不是好方向。

### 8.3 Overall Top 10 Configurations

| Rank | Method | Score | Recall@50 | QPS | Budget | ef | alpha | label_dim_ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `hnsw-filter-aug` | 95.5780 | 0.9267 | 11128.45 | 500 | 400 | 0.9 | 0.01 |
| 2 | `hnsw-filter-aug` | 93.5147 | 0.9267 | 10889.20 | 500 | 200 | 0.9 | 0.01 |
| 3 | `hnsw-filter-aug` | 93.1164 | 0.9578 | 10149.73 | 500 | 200 | 0.7 | 0.01 |
| 4 | `hnsw-filter-aug` | 92.8133 | 0.9579 | 10116.02 | 500 | 400 | 0.7 | 0.01 |
| 5 | `hnsw-filter-aug` | 87.2961 | 0.9746 | 9191.50 | 500 | 200 | 0.5 | 0.01 |
| 6 | `hnsw-filter-aug` | 85.3236 | 0.9678 | 9108.86 | 500 | 200 | 0.9 | 0.05 |
| 7 | `hnsw-filter-aug` | 84.8629 | 0.9678 | 9060.46 | 500 | 400 | 0.9 | 0.05 |
| 8 | `hnsw-filter-aug` | 81.1944 | 0.9842 | 8382.32 | 500 | 400 | 0.3 | 0.01 |
| 9 | `hnsw-filter-aug` | 81.0417 | 0.9746 | 8532.58 | 500 | 400 | 0.5 | 0.01 |
| 10 | `hnsw-filter-aug` | 78.1756 | 0.9630 | 8429.33 | 500 | 800 | 0.7 | 0.01 |

解讀：

- Top 10 全部是 `hnsw-filter-aug`。
- `candidate_budget=500` 在 top configs 中完全主導。
- `label_dim_ratio=0.01` 是最佳區域。
- `ef=200` 和 `ef=400` 都有強結果；`ef=800` 開始因 QPS 下降而不划算。

## 9. v1 參數效應分析

### 9.1 Candidate Budget

`hnsw-filter-aug` grouped by candidate budget：

| Candidate Budget | Runs | Best Score | Mean Score | Best Recall | Mean QPS | Mean Surviving Candidates |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 36 | 95.5780 | 70.9999 | 0.9915 | 7537.13 | 267.46 |
| 1000 | 38 | 64.3643 | 44.6235 | 0.9953 | 4571.37 | 520.25 |
| 2000 | 36 | 36.3019 | 25.0492 | 0.9986 | 2535.34 | 999.16 |

結論：

- 增加 budget 可以提高 recall，但 QPS 下降更大。
- 在目前 score 公式下，`budget=500` 最佳。
- `budget=2000` recall 最高，但 QPS 太低，score 不佳。

### 9.2 `hnsw_alpha` 與 `hnsw_label_dim_ratio`

`hnsw-filter-aug` top parameter regions：

| alpha | label_dim_ratio | Runs | Best Score | Mean Score | Best Recall | Best QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.01 | 9 | 95.5780 | 61.3131 | 0.9888 | 11128.45 |
| 0.7 | 0.01 | 9 | 93.1164 | 59.7820 | 0.9926 | 10149.73 |
| 0.5 | 0.01 | 9 | 87.2961 | 54.8061 | 0.9948 | 9191.50 |
| 0.9 | 0.05 | 9 | 85.3236 | 53.2219 | 0.9939 | 9108.86 |
| 0.3 | 0.01 | 9 | 81.1944 | 49.5039 | 0.9964 | 8382.32 |
| 0.9 | 0.10 | 9 | 76.7590 | 46.4959 | 0.9955 | 8003.04 |

結論：

- `label_dim_ratio=0.01` 最穩定，是 v2 繼續細搜的中心。
- 高 `alpha` 表示仍以原始 vector distance 為主，只加入少量 label signal。
- label 維度太大時，candidate pool 對 label midpoint 過度敏感，會犧牲 vector neighborhood，或增加 graph search 成本。

### 9.3 `ef_search`

`hnsw-filter-aug` grouped by `hnsw_ef_search`：

| ef_search | Runs | Best Score | Mean Score | Mean Recall | Mean QPS |
|---:|---:|---:|---:|---:|---:|
| 200 | 36 | 93.5147 | 48.1292 | 0.9843 | 5019.18 |
| 400 | 38 | 95.5780 | 47.5611 | 0.9848 | 4954.02 |
| 800 | 36 | 78.1756 | 44.8191 | 0.9853 | 4649.38 |

結論：

- `ef=200` 和 `ef=400` 表現接近。
- `ef=400` 取得最高 single-run score。
- `ef=800` recall 只小幅增加，但 QPS 下滑，因此 score 較低。

## 10. 方法差異與取捨

### 10.1 為什麼 `hnsw-filter-aug` 贏

Global HNSW 只依 vector distance 找 candidates。當 label filter 丟掉一部分 candidates 時，effective candidate pool 變小，recall 受影響。

Filter-augmented HNSW 在建 graph 時把 label signal 放入 index space，使 candidates 更容易落在 query filter range 附近。這帶來兩個好處：

1. 同樣 budget 下，label filter 後的 surviving candidates 更有用。
2. 可以把 candidate budget 從 global HNSW 的 1000 降到 v2 best 的 300，QPS 大幅上升。

v2 multi-seed mean best config 的 average surviving candidates：

```text
avg_surviving_candidates = 98.19
avg_survival_rate ≈ 0.3273
queries_with_zero_surviving = 1 across 3 seeds
```

雖然 mean recall 只有 0.9091，但 mean QPS 達 13978.43，使 multi-seed mean score 達 115.5122。若需要較高 recall，v2 也找到 `alpha=0.65`, `label_dim_ratio=0.02`, `budget=300`, `ef=200` 的高 recall config，mean recall 約 0.9550，mean score 仍有 104.1879。

### 10.2 為什麼 `adaptive-hnsw-aug` recall 最高但 score 較低

`adaptive-hnsw-aug` 對 small filters 使用 exact label-sorted search，對 medium filters 使用 augmented HNSW，對 large filters 使用 global HNSW。因此它能維持非常高 recall。

但成本包括：

- 需要兩個 HNSW indexes。
- small exact branch 雖精準但不一定最快。
- routing 後分批查詢可能降低 batch efficiency。

因此 best recall 達 0.9981，但 QPS 只有 5911.19，score 58.8848。

### 10.3 為什麼 dynamic budget 失敗

Dynamic budget 的目標是避免每個 query 都取大 candidate pool。但實作上它需要對同一 query 重複呼叫：

```python
hnsw.knn_query(query, k=current_budget)
```

如果第一次 budget 不夠，就再查一次更大的 k。這會讓 HNSW traversal 成本重複發生，且 Python loop 也破壞 batch query 優勢。

實驗結果：

| Method | Best Score | Recall@50 | QPS |
|---|---:|---:|---:|
| `hnsw-dynamic` | 2.9466 | 0.9703 | 313.00 |
| `hnsw` | 63.6829 | 0.9703 | 6764.14 |

在 recall 幾乎相同的情況下，dynamic QPS 低了超過 20 倍，因此不適合保留在主要 sweep。

## 11. Ablation v2 設計與結果

v2 的設計目標是檢查 v1 best region 是否對 seed 穩定，並針對 best score 附近做細搜。v2 已完成 full SIFT1M 實驗，結果檔為 `hnsw_ablation_v2_full.csv`。

### 11.1 v2 搜尋範圍

| Parameter | Values |
|---|---|
| seeds | `40,41,42` |
| method | `hnsw-filter-aug-v2` |
| `hnsw_alpha` | `0.65, 0.75, 0.85, 0.90, 0.95` |
| `hnsw_label_dim_ratio` | `0.005, 0.01, 0.02` |
| `hnsw_ef_search` | `200, 300, 400, 500` |
| `candidate_budget` | `300, 400, 500, 600, 800` |
| `hnsw_m` | `16` |
| `hnsw_ef_construction` | `200` |

總數：

```text
300 configs per seed
3 seeds
900 logged rows
15 graph builds per seed
45 graph builds total
```

實際 CSV 檢查：

| Item | Value |
|---|---:|
| rows | 900 |
| seeds | 40, 41, 42 |
| rows per seed | 300 |
| graph groups per seed | 15 |
| full SIFT base vectors | 1,000,000 |
| queries | 10,000 |

### 11.2 v2 為什麼先不掃 `M` 和 `ef_construction`

`M` 和 `ef_construction` 主要影響 graph build quality 和 memory/build cost。因為 v1 已經找到 score 很高的 query-time region，v2 先固定：

```text
M = 16
ef_construction = 200
```

把搜尋集中在：

- seed stability
- candidate budget
- query-time ef
- alpha
- label dimension

這樣可以用 graph reuse 降低成本，先確認 best config 是否穩定，再決定是否需要 v3 搜 `M` 和 `ef_construction`。

### 11.3 v2 Overall Best

v2 單次最高分：

| Rank | Method | Seed | Score | Recall@50 | QPS | Budget | ef | alpha | label_dim_ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `hnsw-filter-aug-v2` | 40 | 118.0396 | 0.9101 | 14251.00 | 300 | 200 | 0.65 | 0.005 |
| 2 | `hnsw-filter-aug-v2` | 41 | 116.1479 | 0.8845 | 14844.87 | 300 | 200 | 0.75 | 0.010 |
| 3 | `hnsw-filter-aug-v2` | 41 | 116.0023 | 0.9093 | 14029.40 | 300 | 200 | 0.65 | 0.010 |
| 4 | `hnsw-filter-aug-v2` | 42 | 115.9966 | 0.9077 | 14078.45 | 300 | 200 | 0.65 | 0.010 |
| 5 | `hnsw-filter-aug-v2` | 40 | 115.5120 | 0.8866 | 14694.74 | 300 | 200 | 0.75 | 0.005 |

單次最高不一定是建議提交 config，因為 seed 40 的 `label_dim_ratio=0.005` 分數很高，但 seeds 41/42 下 QPS 較低，跨 seed 平均只有 111.5091。助教若會改 seed，應以 multi-seed mean ranking 為主。

### 11.4 Multi-seed Aggregate Ranking

以 `(alpha, label_dim_ratio, ef_search, candidate_budget)` 分組，對 seeds 40/41/42 取平均：

| Rank | alpha | label_dim_ratio | ef | budget | Mean Score | Std Score | Min Score | Max Score | Mean Recall | Mean QPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.65 | 0.010 | 200 | 300 | 115.5122 | 0.8439 | 114.5378 | 116.0023 | 0.9091 | 13978.43 |
| 2 | 0.75 | 0.010 | 200 | 300 | 113.8763 | 3.1651 | 110.2610 | 116.1479 | 0.8847 | 14547.98 |
| 3 | 0.65 | 0.005 | 200 | 300 | 111.5091 | 6.0321 | 106.1459 | 118.0396 | 0.9090 | 13494.37 |
| 4 | 0.85 | 0.010 | 200 | 300 | 111.2195 | 3.0726 | 107.6847 | 113.2512 | 0.8546 | 15225.91 |
| 5 | 0.85 | 0.020 | 200 | 300 | 111.0760 | 3.5317 | 107.0345 | 113.5689 | 0.9082 | 13466.40 |
| 6 | 0.75 | 0.020 | 200 | 300 | 110.2749 | 1.2312 | 108.9117 | 111.3057 | 0.9375 | 12545.86 |
| 7 | 0.90 | 0.020 | 200 | 300 | 107.7228 | 5.9268 | 100.9067 | 111.6619 | 0.8847 | 13762.00 |
| 8 | 0.85 | 0.005 | 200 | 300 | 106.2156 | 10.7553 | 93.8068 | 112.8613 | 0.8546 | 14538.73 |
| 9 | 0.65 | 0.010 | 300 | 300 | 106.0860 | 3.3208 | 102.3956 | 108.8332 | 0.9091 | 12838.01 |
| 10 | 0.75 | 0.005 | 200 | 300 | 105.9974 | 11.1241 | 93.7668 | 115.5120 | 0.8848 | 13536.82 |

最穩定的 best config 是 rank 1：

| Seed | Score | Recall@50 | QPS | Avg Surviving Candidates |
|---:|---:|---:|---:|---:|
| 40 | 114.5378 | 0.9101 | 13827.44 | 100.47 |
| 41 | 116.0023 | 0.9093 | 14029.40 | 95.92 |
| 42 | 115.9966 | 0.9077 | 14078.45 | 98.18 |
| mean | 115.5122 | 0.9091 | 13978.43 | 98.19 |

解讀：

- `budget=300` 主導 top rankings，代表 v2 成功找到更偏 QPS 的最佳區域。
- `ef=200` 在 top 10 中完全主導。更大的 ef 對 recall 幫助有限，但降低 QPS。
- `alpha=0.65` 是最佳穩定點，平衡 vector similarity 與 label awareness。
- `label_dim_ratio=0.01` 比 `0.005` 更穩定，雖然 `0.005` 有單次最高分。

### 11.5 v1 Best Config in v2 Seeds

v1 best config 是：

```text
alpha=0.9
label_dim_ratio=0.01
ef=400
budget=500
```

在 v2 的 seeds 40/41/42 上重新評估：

| Seed | Score | Recall@50 | QPS |
|---:|---:|---:|---:|
| 40 | 89.9692 | 0.9296 | 10410.81 |
| 41 | 88.7570 | 0.9285 | 10295.03 |
| 42 | 87.6663 | 0.9269 | 10203.30 |
| mean | 88.7975 | 0.9284 | 10303.05 |

與 v2 mean best 比較：

| Config | Mean Score | Mean Recall | Mean QPS |
|---|---:|---:|---:|
| v2 mean best: `alpha=0.65`, `ratio=0.01`, `ef=200`, `budget=300` | 115.5122 | 0.9091 | 13978.43 |
| v1 best re-evaluated: `alpha=0.9`, `ratio=0.01`, `ef=400`, `budget=500` | 88.7975 | 0.9284 | 10303.05 |

v2 best 平均 score 比 v1 best re-evaluated 高約 30.1%。原因不是 recall 更高，而是 QPS 高出約 35.7%。在目前 score 公式下，這個 QPS gain 足以抵消 recall 下降。

### 11.6 v2 Parameter Effects

#### Candidate Budget

| Budget | Runs | Mean Score | Max Score | Mean Recall | Mean QPS | Mean Surviving Candidates | Zero-surviving Queries |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 180 | 95.8114 | 118.0396 | 0.8806 | 12409.91 | 86.80 | 54 |
| 400 | 180 | 93.4057 | 108.2335 | 0.9232 | 10992.54 | 114.27 | 41 |
| 500 | 180 | 85.7276 | 96.8674 | 0.9472 | 9575.28 | 141.49 | 28 |
| 600 | 180 | 76.8289 | 88.6602 | 0.9617 | 8321.26 | 168.58 | 20 |
| 800 | 180 | 64.9502 | 75.0859 | 0.9759 | 6827.55 | 222.19 | 8 |

結論：

- budget 越大 recall 越高，但 QPS 快速下降。
- v2 的 score 最佳點在 `budget=300`。
- 若任務有更高 recall 要求，可以選 `budget=400/500`，但 score 會下降。

#### `ef_search`

| ef_search | Runs | Mean Score | Max Score | Mean Recall | Mean QPS |
|---:|---:|---:|---:|---:|---:|
| 200 | 225 | 85.9886 | 118.0396 | 0.9366 | 9983.74 |
| 300 | 225 | 84.4575 | 115.3524 | 0.9366 | 9793.42 |
| 400 | 225 | 82.5789 | 105.7794 | 0.9381 | 9518.75 |
| 500 | 225 | 80.3540 | 99.4093 | 0.9395 | 9205.31 |

結論：

- `ef_search` 增加只有非常小的 recall gain。
- QPS 下降更明顯，因此 `ef=200` 是 v2 最佳選擇。

#### `hnsw_alpha`

| alpha | Runs | Mean Score | Max Score | Mean Recall | Mean QPS |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 180 | 83.3426 | 118.0396 | 0.9632 | 9071.02 |
| 0.75 | 180 | 84.0991 | 116.1479 | 0.9521 | 9389.34 |
| 0.85 | 180 | 83.6540 | 113.5689 | 0.9362 | 9698.46 |
| 0.90 | 180 | 79.9642 | 111.6619 | 0.9255 | 9477.35 |
| 0.95 | 180 | 85.6639 | 107.9403 | 0.9117 | 10490.36 |

單看 alpha 的平均，`0.95` 不差，因為 QPS 高；但 top multi-seed config 是 `0.65`，代表最佳點不是單純越接近 pure vector 越好，而是需要足夠 label signal 讓小 candidate budget 仍保有可用 survivors。

#### `hnsw_label_dim_ratio`

| label_dim_ratio | Runs | Mean Score | Max Score | Mean Recall | Mean QPS |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 300 | 83.8209 | 118.0396 | 0.9296 | 9845.51 |
| 0.010 | 300 | 86.4819 | 116.1479 | 0.9296 | 10160.70 |
| 0.020 | 300 | 79.7315 | 113.5689 | 0.9540 | 8869.71 |

結論：

- `0.01` 是最佳平均 score。
- `0.005` 可以跑出最高 single-run score，但不如 `0.01` 穩定。
- `0.02` recall 高，但 QPS 低；它比較適合 high-recall config。

### 11.7 High-recall Alternatives

如果助教或評分者希望 recall 不要降到 0.91 左右，可以選擇以下 v2 high-recall alternatives：

| Recall Requirement | alpha | label_dim_ratio | ef | budget | Mean Score | Mean Recall | Mean QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| recall >= 0.95 | 0.65 | 0.020 | 200 | 300 | 104.1879 | 0.9550 | 11423.94 |
| recall >= 0.97 | 0.65 | 0.020 | 300 | 400 | 89.2400 | 0.9730 | 9426.16 |
| highest recall in v2 grid | 0.65 | 0.020 | 300 | 800 | 55.5641 | 0.9895 | 5674.35 |

這些結果顯示 recall 和 score 的 trade-off 很清楚：若把 recall 拉高到接近 0.99，QPS 會大幅下降，final score 反而不如最佳 score config。

### 11.8 v2 Runtime Notes

v2 runner 的 `notes` 欄位記錄了 exact ground truth 與 graph build time。三個 seed 的 exact ground truth time 約：

| Seed | Exact Ground Truth Time |
|---:|---:|
| 40 | 871.144 s |
| 41 | 884.947 s |
| 42 | 1085.893 s |

每個 `(seed, alpha, label_dim_ratio)` graph build time 平均約 18.52 s，共 45 個 graph groups。這證明 graph reuse 有實際價值：若每個 config 都重建 graph，v2 會需要 900 次 graph build；現在只需要 45 次。

## 12. 重現指令

### 12.1 Best Config

建議重現 v2 multi-seed mean best。注意：`hnsw-filter-aug-v2` 是 v2 runner 的 CSV method name；單次重現請使用 `main.py --method hnsw-filter-aug`。

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

### 12.2 v1 Full Ablation

```bash
RESULTS_CSV=experiments/hnsw_ablation_v1_full.csv \
LOG_FILE=experiments/hnsw_ablation_v1_full.log \
bash run_hnsw_ablation_uv.sh --sift --max-base 1000000 --n-query 10000
```

### 12.3 v2 Full Ablation

```bash
RESULTS_CSV=experiments/hnsw_ablation_v2_full.csv \
LOG_FILE=experiments/hnsw_ablation_v2_full.log \
SEEDS=40,41,42 \
bash run_hnsw_ablation_v2_uv.sh --sift --max-base 1000000 --n-query 10000
```

## 13. 限制與未來工作

### 13.1 目前限制

- v2 已經用 seeds 40/41/42 驗證穩定性，但更多 seed 仍可降低隨機 label/filter range 的不確定性。
- label 是 synthetic uniform label，不是語意型 metadata。
- filter 目前是 single integer range，尚未測 multi-attribute filters。
- `hnsw-filter-aug` 的最佳 score 來自犧牲部分 recall；如果評分規則改成 strict recall threshold，它不一定是最佳。
- HNSW build time 不計入 QPS；若實際系統需要頻繁更新 index，需另外評估 build/update 成本。

### 13.2 後續可做

1. 在 v2 best config 附近掃 `hnsw_m` 和 `hnsw_ef_construction`。
2. 針對不同 selectivity range 分別找 best config。
3. 嘗試 per-selectivity candidate budget，但必須避免 dynamic method 那種 repeated HNSW query。
4. 改善 batch routing，讓 adaptive method 減少分支後的小 batch overhead。
5. 記錄 memory usage 和 build time，補充實務部署考量。

## 14. 結論

本專案的主要發現是：在 filtered ANN 任務中，單純 global ANN + postfilter 會浪費候選集合；但如果在 graph index 建立階段注入少量 filter/label signal，就可以用更小 candidate budget 取得足夠 recall，並大幅提高 QPS。

`hnsw-filter-aug` 是目前最好的方法，因為它符合本專案評分公式的核心需求：

```text
不要盲目追求 recall = 1.0，而是在 recall 仍可接受時最大化 QPS。
```

目前最佳穩定 config 是 v2 multi-seed mean best：

```text
Final Score = 115.5122
Recall@50 = 0.9091
QPS = 13978.43
candidate_budget = 300
ef_search = 200
hnsw_alpha = 0.65
hnsw_label_dim_ratio = 0.01
```

v2 也確認了 v1 best config 不是 multi-seed 最佳；更小的 candidate budget 和更強的 label-aware graph bias 可以換來更高 QPS，並把平均 final score 從 v1 best re-evaluated 的 88.7975 推到 115.5122。
