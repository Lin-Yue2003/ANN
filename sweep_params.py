"""
sweep_params.py
===============
Generate parameter sweep configurations for experiments.

Supports different sweep scales: baseline, small, medium, full.
"""

import json
from typing import List, Dict, Any


def cli_arg_for_param(key: str) -> str:
    if key == 'n_tables':
        return '--lsh-tables'
    if key == 'n_functions':
        return '--lsh-functions'
    if key == 'bin_width':
        return '--lsh-bin-width'
    return f"--{key.replace('_', '-')}"


def generate_baseline_configs() -> List[Dict[str, Any]]:
    """
    Baseline: current default parameters.
    """
    return [
        {
            'method': 'postfilter',
            'alpha': 0.05,
            'label_dim_ratio': 0.05,
            'n_tables': 400,
            'n_functions': 5,
            'bin_width': 0.22,
        },
    ]


def generate_small_sweep_configs() -> List[Dict[str, Any]]:
    """
    Small sweep: limited parameter grid for quick validation (~50 configs).
    """
    configs = []
    
    # Base postfilter variations
    for alpha in [0.01, 0.05, 0.1, 0.3]:
        for n_tables in [100, 200, 400]:
            for bin_width in [0.1, 0.22, 0.5]:
                configs.append({
                    'method': 'postfilter',
                    'alpha': alpha,
                    'label_dim_ratio': 0.05,
                    'n_tables': n_tables,
                    'n_functions': 5,
                    'bin_width': bin_width,
                })
    
    # Adaptive variations
    for tau_small in [0.01, 0.05]:
        for tau_medium in [0.1, 0.2]:
            configs.append({
                'method': 'adaptive',
                'alpha': 0.05,
                'label_dim_ratio': 0.05,
                'n_tables': 200,
                'n_functions': 5,
                'bin_width': 0.22,
                'tau_small': tau_small,
                'tau_medium': tau_medium,
            })

    # HNSW candidate-generation variations
    for ef_search in [100, 200, 400]:
        for candidate_budget in [200, 500, 1000]:
            configs.append({
                'method': 'hnsw',
                'hnsw_m': 16,
                'hnsw_ef_construction': 200,
                'hnsw_ef_search': ef_search,
                'candidate_budget': candidate_budget,
            })

    # Adaptive HNSW variations
    for tau_small in [0.01, 0.03]:
        for tau_medium in [0.1, 0.2]:
            for medium_index in ['hnsw', 'lsh']:
                configs.append({
                    'method': 'adaptive-hnsw',
                    'tau_small': tau_small,
                    'tau_medium': tau_medium,
                    'adaptive_medium_index': medium_index,
                    'hnsw_m': 16,
                    'hnsw_ef_construction': 200,
                    'hnsw_ef_search': 200,
                    'candidate_budget': 1000,
                    'alpha': 0.05,
                    'label_dim_ratio': 0.05,
                    'n_tables': 200,
                    'n_functions': 5,
                    'bin_width': 0.22,
                })
    
    return configs


def generate_full_sweep_configs() -> List[Dict[str, Any]]:
    """
    Full sweep: comprehensive grid as per prompt.txt (~500+ configs).
    """
    configs = []
    
    # ===== Postfilter with full parameter sweep =====
    alphas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8]
    label_dim_ratios = [0.01, 0.03, 0.05, 0.1, 0.2]
    n_tables_list = [100, 200, 300, 400, 600, 800]
    n_functions_list = [2, 3, 4, 5]
    bin_widths = [0.10, 0.15, 0.20, 0.22, 0.30, 0.40, 0.60, 0.80, 1.00]
    
    for alpha in alphas:
        for label_dim_ratio in label_dim_ratios:
            for n_tables in n_tables_list:
                for n_functions in n_functions_list:
                    for bin_width in bin_widths:
                        configs.append({
                            'method': 'postfilter',
                            'alpha': alpha,
                            'label_dim_ratio': label_dim_ratio,
                            'n_tables': n_tables,
                            'n_functions': n_functions,
                            'bin_width': bin_width,
                        })
    
    # ===== Adaptive search with selectivity thresholds =====
    tau_smalls = [0.005, 0.01, 0.02, 0.03, 0.05]
    tau_mediums = [0.05, 0.10, 0.15, 0.20, 0.30]
    
    for tau_small in tau_smalls:
        for tau_medium in tau_mediums:
            if tau_small < tau_medium:  # Only valid combinations
                configs.append({
                    'method': 'adaptive',
                    'alpha': 0.05,
                    'label_dim_ratio': 0.05,
                    'n_tables': 300,
                    'n_functions': 4,
                    'bin_width': 0.22,
                    'tau_small': tau_small,
                    'tau_medium': tau_medium,
                })

    # ===== Global HNSW post-filter =====
    hnsw_ms = [8, 16, 32]
    ef_constructions = [100, 200, 400]
    ef_searches = [100, 200, 400, 800]
    candidate_budgets = [100, 200, 500, 1000, 2000]

    for hnsw_m in hnsw_ms:
        for ef_construction in ef_constructions:
            for ef_search in ef_searches:
                for candidate_budget in candidate_budgets:
                    configs.append({
                        'method': 'hnsw',
                        'hnsw_m': hnsw_m,
                        'hnsw_ef_construction': ef_construction,
                        'hnsw_ef_search': ef_search,
                        'candidate_budget': candidate_budget,
                    })

    # ===== Adaptive HNSW with exact small-range branch =====
    for tau_small in tau_smalls:
        for tau_medium in tau_mediums:
            if tau_small >= tau_medium:
                continue
            for medium_index in ['hnsw', 'lsh']:
                for ef_search in [200, 400, 800]:
                    for candidate_budget in [500, 1000, 2000]:
                        configs.append({
                            'method': 'adaptive-hnsw',
                            'tau_small': tau_small,
                            'tau_medium': tau_medium,
                            'adaptive_medium_index': medium_index,
                            'hnsw_m': 16,
                            'hnsw_ef_construction': 200,
                            'hnsw_ef_search': ef_search,
                            'candidate_budget': candidate_budget,
                            'alpha': 0.05,
                            'label_dim_ratio': 0.05,
                            'n_tables': 300,
                            'n_functions': 4,
                            'bin_width': 0.22,
                        })
    
    return configs


def generate_shell_commands(configs: List[Dict[str, Any]], 
                           base_args: str = "",
                           output_log: str = "experiments/results.csv") -> List[str]:
    """
    Convert config list to shell commands.
    
    Parameters
    ----------
    configs : list of dicts
        Configuration dictionaries
    base_args : str
        Common arguments (e.g., "--sift --sift-dir ./data")
    output_log : str
        Output CSV file path
    
    Returns
    -------
    List of shell command strings
    """
    commands = []
    for i, cfg in enumerate(configs):
        cmd_parts = [f"python main.py {base_args}"]
        cmd_parts.append(f"--method {cfg['method']}")
        
        # Add all hyperparameters
        for key, value in cfg.items():
            if key != 'method':
                cmd_parts.append(f"{cli_arg_for_param(key)} {value}")
        
        cmd_parts.append(f"--experiment-log {output_log}")
        cmd_parts.append(f"# Config {i+1}/{len(configs)}")
        
        commands.append(" ".join(cmd_parts))
    
    return commands


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sweep_params.py {baseline|small|full} [base_args]")
        print("  base_args: e.g., '--sift --sift-dir ./data'")
        sys.exit(1)
    
    sweep_type = sys.argv[1]
    base_args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    
    if sweep_type == 'baseline':
        configs = generate_baseline_configs()
    elif sweep_type == 'small':
        configs = generate_small_sweep_configs()
    elif sweep_type == 'full':
        configs = generate_full_sweep_configs()
    else:
        print(f"Unknown sweep type: {sweep_type}")
        sys.exit(1)
    
    commands = generate_shell_commands(configs, base_args)
    
    print(f"# Generated {len(commands)} configurations")
    print()
    for cmd in commands:
        print(cmd)
