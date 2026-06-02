"""
experiments/logger.py
=====================
Experiment results logger for CSV output.

Records all experimental runs with hyperparameters and metrics.
"""

import csv
import time
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    CSV logger for filtered ANNS experiments.
    
    Writes one row per experiment with:
    - Metadata: timestamp, seed, dataset config
    - Hyperparameters: all method-specific parameters
    - Metrics: recall, QPS, final score, candidate statistics
    """
    
    FIELDNAMES = [
        'timestamp',
        'method',
        'seed',
        'dataset_mode',
        'n_base',
        'n_query',
        'k',
        'n_labels',
        'min_sel',
        'max_sel',
        'mean_sel',
        # Hyperparameters
        'alpha',
        'label_dim_ratio',
        'n_tables',
        'n_functions',
        'bin_width',
        'tau_small',
        'tau_medium',
        'adaptive_medium_index',
        'hnsw_m',
        'hnsw_ef_construction',
        'hnsw_ef_search',
        'candidate_budget',
        'initial_candidate_budget',
        'budget_expansion_factor',
        'min_survivors_multiplier',
        'hnsw_alpha',
        'hnsw_label_dim_ratio',
        'adaptive_budget_tau_small',
        'adaptive_budget_tau_medium',
        'adaptive_budget_small',
        'adaptive_budget_medium',
        'adaptive_budget_large',
        'hnsw_n_shards',
        'shard_min_budget',
        'probe_radius',
        # Metrics
        'search_time_s',
        'qps',
        'mean_recall',
        'median_recall',
        'min_recall',
        'max_recall',
        'final_score',
        # Candidate stats
        'avg_total_candidates',
        'avg_surviving_candidates',
        'avg_survival_rate',
        'queries_with_zero_candidates',
        'queries_with_zero_surviving',
        'recall_sel_bin_000_005',
        'recall_sel_bin_005_010',
        'recall_sel_bin_010_020',
        'recall_sel_bin_020_050',
        'recall_sel_bin_050_100',
        'recall_sel_bin_100_200',
        'recall_sel_bin_200_plus',
        'notes',
    ]
    
    def __init__(self, log_file: str = 'experiments/results.csv'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists to avoid re-writing header
        self.file_exists = self.log_file.exists()
    
    def log_experiment(self, 
                       method: str,
                       metrics: Dict[str, Any],
                       hyperparams: Dict[str, Any],
                       dataset_config: Dict[str, Any],
                       notes: str = ""):
        """
        Log one experiment to CSV.
        
        Parameters
        ----------
        method : str
            Method name (e.g., "postfilter", "adaptive", "label_sorted")
        metrics : dict
            Keys: search_time_s, qps, mean_recall, median_recall, min_recall, 
                  max_recall, final_score, candidate statistics
        hyperparams : dict
            Method-specific hyperparameters
        dataset_config : dict
            Keys: n_base, n_query, k, n_labels, min_sel, max_sel, mean_sel, seed, dataset_mode
        notes : str
            Optional notes about the run
        """
        row = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': method,
            'notes': notes,
        }
        
        # Dataset config
        row.update(dataset_config)
        
        # Hyperparameters (fill in None for unused ones)
        for fname in self.FIELDNAMES:
            if fname.startswith('recall_sel_bin_'):
                continue
            if fname not in row:
                row[fname] = hyperparams.get(fname, '')
        
        # Metrics
        row.update(metrics)
        
        # Selectivity bins - extract from metrics if present
        if 'recall_by_sel_bins' in metrics:
            bins_dict = metrics['recall_by_sel_bins']
            for i, (bin_name, recall) in enumerate(sorted(bins_dict.items())):
                csv_col = f'recall_sel_bin_{bin_name}'
                if csv_col in self.FIELDNAMES:
                    row[csv_col] = recall
        
        # Write header if file is new
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if not self.file_exists or f.tell() == 0:
                writer.writeheader()
                self.file_exists = True
            writer.writerow(row)
        
        print(f"[Logger] Logged to {self.log_file}")


def compute_selectivity_bins(selectivities):
    """
    Compute recall statistics binned by selectivity percentiles.
    
    Parameters
    ----------
    selectivities : (Q,) array of float
        Selectivity for each query
    
    Returns
    -------
    dict with bin boundaries as keys
    """
    # Define bins by percentile ranges (as fractions of full label space)
    # 0-0.5%, 0.5-1%, 1-2%, 2-5%, 5-10%, 10-20%, 20%+
    bins = [
        (0.000, 0.005, '000_005'),
        (0.005, 0.010, '005_010'),
        (0.010, 0.020, '010_020'),
        (0.020, 0.050, '020_050'),
        (0.050, 0.100, '050_100'),
        (0.100, 0.200, '100_200'),
        (0.200, 1.000, '200_plus'),
    ]
    
    result = {}
    for lo, hi, name in bins:
        mask = (selectivities >= lo) & (selectivities < hi)
        if mask.sum() > 0:
            result[name] = mask.sum()  # Will be updated with actual recalls
        else:
            result[name] = 0
    
    return result
