#!/bin/bash
# REMOTE_EXECUTION_GUIDE.sh
# =========================
# 快速參考：如何在遠端機器上執行優化框架

cat << 'EOF'

╔═════════════════════════════════════════════════════════════╗
║  Filtered ANNS 優化框架 - 遠端執行指南                    ║
╚═════════════════════════════════════════════════════════════╝

📋 前置準備 (本機)
═══════════════════════════════════════════════════════════════

1. 上傳項目到遠端機器
   $ scp -r /Users/yue/巨量資料分析/final/term_project/ \
       remote-user@remote-host:/path/to/filtering-anns/

2. 連接遠端機器
   $ ssh remote-user@remote-host
   $ cd /path/to/filtering-anns/


🚀 在遠端機器上執行
═══════════════════════════════════════════════════════════════

選項 1：快速驗證 (1-2 分鐘) ⚡
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ bash run_all.sh baseline

結果位置: experiments/results.csv
檢查: tail experiments/results.csv


選項 2：中等規模掃描 (20-40 分鐘) 📊
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ bash run_all.sh small

結果位置: experiments/results.csv
進度監控: tail -f experiments/sweep_*.log


選項 3：完整 SIFT 優化 (8-16 小時) 🏁
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

前置要求：需要 SIFT1M 數據
$ ls data/sift_base.fvecs data/sift_query.fvecs

若數據不存在，下載：
$ wget -P data/ http://corpus-texmex.irisa.fr/sift_base.fvecs.gz
$ wget -P data/ http://corpus-texmex.irisa.fr/sift_query.fvecs.gz
$ gunzip data/*.gz

執行完整掃描：
$ bash run_all.sh full

結果位置: experiments/results_full_sift.csv
進度監控: tail -f experiments/sweep_*.log


選項 4：漸進式執行 (所有層級順序)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ bash run_all.sh progressive

步驟：
  1. Baseline (1-2 min)    ✓
  2. Small sweep (20-40 min)     ✓
  3. Full sweep (8-16 hours)     ✓


💾 監控和收集結果
═══════════════════════════════════════════════════════════════

實時監控進度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ watch -n 5 'wc -l experiments/results.csv'
$ tail -f experiments/sweep_*.log

下載結果 (在本機上執行)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ scp -r remote-user@remote-host:/path/to/filtering-anns/experiments/ \
        ./experiments_remote/


📈 分析結果 (本機)
═══════════════════════════════════════════════════════════════

自動分析最佳配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ cd /Users/yue/巨量資料分析/final/term_project/
$ python analyze_results.py experiments_remote/results.csv

輸出示例：
  Total experiments: 142
  Methods tested: postfilter, adaptive
  
  OVERALL BEST CONFIGURATION
  ═════════════════════════════
  Method:        adaptive
  Final Score:   0.3456
  Mean Recall:   0.5812
  QPS:           10234.5
  
  Hyperparameters:
    alpha              = 0.05
    label_dim_ratio    = 0.05
    n_tables           = 300
    n_functions        = 4
    bin_width          = 0.22
    tau_small          = 0.01
    tau_medium         = 0.15


🔍 檢查和調試
═══════════════════════════════════════════════════════════════

驗證框架是否完整
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ bash quick_test.sh
# 預期：✓ Framework test PASSED (30 秒內完成)

生成參數命令
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ python sweep_params.py baseline       # 1 個配置
$ python sweep_params.py small          # ~50 個配置
$ python sweep_params.py full           # ~550+ 個配置

自訂單個實驗
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
$ python main.py \
    --max-base 50000 \
    --n-query 500 \
    --method adaptive \
    --alpha 0.1 \
    --n-tables 300 \
    --tau-small 0.01 \
    --tau-medium 0.20 \
    --experiment-log experiments/custom.csv


⏱️ 預期執行時間
═════════════════════════════════════════════════════════════════

| 層級 | 配置數 | 數據集 | 預期時間 |
|------|--------|--------|----------|
| Baseline | 1 | 10K base, 200Q | 1-2 min |
| Small | ~50 | 50K base, 500Q | 20-40 min |
| Full | ~550+ | SIFT1M | 8-16 hours |


📄 輸出文件
═════════════════════════════════════════════════════════════════

主要結果
  experiments/results.csv           - 所有實驗結果 (Append 模式)
  experiments/results_full_sift.csv - 完整 SIFT 結果 (如執行 full)

日誌
  experiments/sweep_TIMESTAMP.log   - 詳細執行日誌

分析
  python analyze_results.py <csv>   - 自動生成分析報告


❗ 常見問題
═════════════════════════════════════════════════════════════════

Q: 腳本許可權被拒
A: $ chmod +x run_all.sh run_baseline.sh run_small_sweep.sh ...

Q: SIFT 數據未找到
A: 檢查 ./data/ 目錄是否存在 sift_base.fvecs 和 sift_query.fvecs

Q: 執行中斷了怎麼辦
A: 結果已自動保存到 CSV，重新執行會繼續追加

Q: 如何加速執行
A: 編輯 sweep_params.py 減少參數範圍，或運行 small 而非 full


🎯 典型工作流
═════════════════════════════════════════════════════════════════

1️⃣ 快速驗證 (確保設置正確)
   $ bash run_all.sh baseline
   ⏱ 1-2 分鐘

2️⃣ 下載並檢查結果
   $ scp -r remote:/path/experiments/ ./
   $ python analyze_results.py experiments/results.csv

3️⃣ 若結果滿意，執行完整掃描
   $ bash run_all.sh full
   ⏱ 8-16 小時 (可在後台運行)

4️⃣ 最終分析和驗證
   $ python analyze_results.py experiments/results_full_sift.csv


💡 提示
═════════════════════════════════════════════════════════════════

• 建議在 tmux/screen 中運行以支援會話斷開重連
• 可用 & 符號在後台運行，如: nohup bash run_all.sh full &
• 監控磁盤空間，CSV 最終可能達到 10-100 MB
• 結果自動 append，不會丟失已有數據


════════════════════════════════════════════════════════════════

更多詳情請見：
  FRAMEWORK.md                - 完整使用指南
  IMPLEMENTATION_SUMMARY.md   - 實施細節

EOF
