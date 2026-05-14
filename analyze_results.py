"""
analyze_results.py
==================
Analyze experiment results CSV and generate summary statistics.

Usage:
  python analyze_results.py experiments/results.csv
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
import statistics


def analyze_results(csv_file):
    """Analyze and print experiment results summary."""
    
    if not Path(csv_file).exists():
        print(f"ERROR: File not found: {csv_file}")
        return
    
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    if not results:
        print(f"WARNING: No results in {csv_file}")
        return
    
    print("\n" + "=" * 80)
    print(f"  Results Analysis: {csv_file}")
    print("=" * 80)
    print(f"\nTotal experiments: {len(results)}")
    
    # Group by method
    by_method = defaultdict(list)
    for row in results:
        method = row.get('method', 'unknown')
        by_method[method].append(row)
    
    print(f"Methods tested: {', '.join(sorted(by_method.keys()))}")
    print()
    
    # For each method, show best configurations
    for method in sorted(by_method.keys()):
        method_rows = by_method[method]
        
        print("-" * 80)
        print(f"  METHOD: {method.upper()}")
        print("-" * 80)
        
        # Try to extract numeric metrics
        try:
            scores = []
            for row in method_rows:
                try:
                    score = float(row.get('final_score', 0))
                    if score > 0:
                        scores.append((score, row))
                except (ValueError, TypeError):
                    pass
            
            if scores:
                # Sort by final score (descending)
                scores.sort(reverse=True)
                
                # Top 3 configurations
                print(f"\n  Top 3 configurations by final_score:\n")
                print(f"  {'#':<3} {'Score':<10} {'Recall':<10} {'QPS':<10} {'Parameters':<40}")
                print("-" * 80)
                
                for rank, (score, row) in enumerate(scores[:3], 1):
                    recall = row.get('mean_recall', '?')
                    qps = row.get('qps', '?')
                    
                    # Build param string
                    params = []
                    for key in ['alpha', 'n_tables', 'bin_width', 'tau_small', 'tau_medium']:
                        val = row.get(key, '')
                        if val and val != '':
                            try:
                                val = float(val)
                                if val == int(val):
                                    params.append(f"{key}={int(val)}")
                                else:
                                    params.append(f"{key}={val:.3f}")
                            except (ValueError, TypeError):
                                pass
                    param_str = ', '.join(params)[:40]
                    
                    print(f"  {rank:<3} {float(score):<10.4f} {recall:<10} {qps:<10} {param_str:<40}")
                
                # Statistics
                print(f"\n  Statistics across all {len(scores)} configurations:")
                try:
                    score_values = [s[0] for s in scores]
                    print(f"    Mean score:   {statistics.mean(score_values):.4f}")
                    print(f"    Median score: {statistics.median(score_values):.4f}")
                    print(f"    Min score:    {min(score_values):.4f}")
                    print(f"    Max score:    {max(score_values):.4f}")
                    
                    if len(score_values) > 1:
                        try:
                            stdev = statistics.stdev(score_values)
                            print(f"    Std dev:      {stdev:.4f}")
                        except:
                            pass
                except:
                    pass
                
            else:
                print(f"  (No valid scores found)")
        
        except Exception as e:
            print(f"  (Error analyzing: {e})")
        
        print()
    
    # Overall best
    print("=" * 80)
    print("  OVERALL BEST CONFIGURATION")
    print("=" * 80)
    
    all_scores = []
    for row in results:
        try:
            score = float(row.get('final_score', 0))
            if score > 0:
                all_scores.append((score, row))
        except (ValueError, TypeError):
            pass
    
    if all_scores:
        all_scores.sort(reverse=True)
        best_score, best_row = all_scores[0]
        
        print(f"\nMethod:        {best_row.get('method', '?')}")
        print(f"Final Score:   {best_score:.4f}")
        print(f"Mean Recall:   {best_row.get('mean_recall', '?')}")
        print(f"QPS:           {best_row.get('qps', '?')}")
        print(f"Search time:   {best_row.get('search_time_s', '?')} seconds")
        print(f"\nHyperparameters:")
        for key in ['alpha', 'label_dim_ratio', 'n_tables', 'n_functions', 'bin_width', 
                    'tau_small', 'tau_medium', 'probe_radius']:
            val = best_row.get(key, '')
            if val and val != '':
                print(f"  {key:<20} = {val}")
        
        print(f"\nDataset:")
        print(f"  N (base vectors):  {best_row.get('n_base', '?')}")
        print(f"  Q (queries):       {best_row.get('n_query', '?')}")
        print(f"  K (neighbors):     {best_row.get('k', '?')}")
        print(f"  n_labels:          {best_row.get('n_labels', '?')}")
        
        print(f"\nTo reproduce this result:")
        print(f"  python main.py \\")
        print(f"    --method {best_row.get('method', '?')} \\")
        for key in ['alpha', 'label_dim_ratio', 'n_tables', 'n_functions', 'bin_width',
                    'tau_small', 'tau_medium']:
            val = best_row.get(key, '')
            if val and val != '':
                print(f"    --{key.replace('_', '-')} {val} \\")
        if best_row.get('dataset_mode') == 'sift':
            print(f"    --sift --sift-dir ./data \\")
        print(f"    --seed {best_row.get('seed', '42')}")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <csv_file>")
        sys.exit(1)
    
    analyze_results(sys.argv[1])
