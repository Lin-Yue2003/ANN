"""
main.py
=======
Main experiment script for filtered approximate nearest-neighbor search.

Two methods are compared:
  A) Pre-filtering  (exact KNN on filtered subset)  ← brute-force ground truth
  B) Post-filtering (LSH ANN on full set, then filter)

Usage
-----
  # Synthetic data (quick demo, no downloads needed):
  python main.py

  # With SIFT1M (download from http://corpus-texmex.irisa.fr/ first):
  python main.py --sift --sift-dir ./sift --max-base 100000 --n-query 1000

Tuning knobs you can experiment with:
  --n-base, --n-query, --dim        dataset size / dimensionality
  --n-labels                        number of distinct attribute values
  --min-sel, --max-sel              filter selectivity range
  --k                               number of nearest neighbors
  --lsh-tables, --lsh-functions     LSH index parameters
  --lsh-bin-width                   LSH bin width (key recall tuning knob)
  --k-multiplier                    post-filter over-fetch factor (see paper)
"""

import argparse
import sys
import os
import numpy as np
import time

# Make sure our modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from data_utils   import (load_sift, generate_synthetic_data,
                           assign_labels, generate_filter_ranges)
from prefilter    import PreFilterSearch
from postfilter   import PostFilterSearch
from adaptive_search import AdaptiveFilteredSearch
from experiments.logger import ExperimentLogger
from evaluate     import print_comparison, compute_recall, compute_qps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Filtered ANNS: Pre-filtering vs Post-filtering (LSH)")

    # --- Dataset ---
    p.add_argument("--sift",       action="store_true",
                   help="Load SIFT1M instead of synthetic data")
    p.add_argument("--sift-dir",   default="./data",
                   help="Directory containing sift_base.fvecs / sift_query.fvecs")
    p.add_argument("--max-base",   type=int, default=None,
    # p.add_argument("--max-base",   type=int, default=100_000,
                   help="Max base vectors to load (SIFT or synthetic)")
    # p.add_argument("--n-query",    type=int, default=1_000,
    p.add_argument("--n-query",    type=int, default=None,
                   help="Number of query vectors")
    p.add_argument("--dim",        type=int, default=128,
                   help="Vector dimensionality (only for synthetic data)")

    # --- Filter / label ---
    p.add_argument("--n-labels",   type=int,   default=1000,
                   help="Number of distinct integer label values")
    p.add_argument("--min-sel",    type=float, default=0.05,
                   help="Min filter selectivity (fraction of label space)")
    p.add_argument("--max-sel",    type=float, default=0.40,
                   help="Max filter selectivity (fraction of label space)")

    # --- Search ---
    p.add_argument("--k",          type=int, default=50,
                   help="Number of nearest neighbors to retrieve")

    # --- LSH ---
    p.add_argument("--lsh-tables",    type=int,   default=400,
                   help="Number of LSH hash tables (L). "
                        "More tables → higher recall, more memory.")
    p.add_argument("--lsh-functions", type=int,   default=5,
                   help="Hash functions concatenated per table key (K). "
                        "Fewer functions → larger (less specific) buckets, "
                        "higher per-table recall but more false positives. "
                        "For 128-D normalised vectors, K=2 works well.")
    p.add_argument("--lsh-bin-width", type=float, default=0.22,
                   help="LSH bin width (w) – key recall tuning knob. "
                        "For L2-normalised vectors (unit sphere) use ~0.3-0.8; "
                        "for raw SIFT (values up to 255) use ~50-200.")
    p.add_argument("--k-multiplier",  type=int,   default=2,  # not used
                   help="Post-filter over-fetch factor")
    
    # --- Filter-augmented params ---
    p.add_argument("--filter-augmented", action="store_true", default=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--label-dim-ratio", type=float, default=0.05)
    
    # --- Search method selection ---
    p.add_argument("--method", 
                   choices=["postfilter", "adaptive", "label-sorted"],
                   default="postfilter",
                   help="Search method to use")
    
    # --- Adaptive search parameters ---
    p.add_argument("--tau-small", type=float, default=0.01,
                   help="Selectivity threshold for small-range exact search")
    p.add_argument("--tau-medium", type=float, default=0.15,
                   help="Selectivity threshold for medium-range LSH search")
    
    # --- Multi-probe parameters ---
    p.add_argument("--probe-radius", type=int, default=0,
                   help="Multi-probe LSH: probe radius (0=disabled)")
    p.add_argument("--max-extra-probes", type=int, default=0,
                   help="Multi-probe LSH: max extra probes per table")
    
    # --- Logging ---
    p.add_argument("--experiment-log", type=str, default=None,
                   help="CSV file to log experiment results")

    # --- Misc ---
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Filtered ANNS Experiment")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load / generate data
    # ------------------------------------------------------------------
    if args.sift:
        base_vecs, query_vecs = load_sift(
            base_dir  = args.sift_dir,
            max_base  = args.max_base,
            max_query = args.n_query,
        )
    else:
        base_vecs, query_vecs = generate_synthetic_data(
            n_base  = args.max_base,
            n_query = args.n_query,
            dim     = args.dim,
            seed    = args.seed,
        )
        # Trim queries if needed
        query_vecs = query_vecs[:args.n_query]

    N, D = base_vecs.shape
    Q    = len(query_vecs)

    # ------------------------------------------------------------------
    # 2. Assign random labels to base vectors; generate filter ranges
    # ------------------------------------------------------------------
    labels = assign_labels(N, n_labels=args.n_labels, seed=args.seed)
    filter_ranges = generate_filter_ranges(
        Q,
        n_labels        = args.n_labels,
        min_selectivity = args.min_sel,
        max_selectivity = args.max_sel,
        seed            = args.seed + 1,
    )

    selectivities = (filter_ranges[:, 1] - filter_ranges[:, 0] + 1) / args.n_labels
    print(f"\n[config] N={N:,}  Q={Q:,}  D={D}  K={args.k}  n_labels={args.n_labels}")
    print(f"[config] Filter selectivity  "
          f"mean={selectivities.mean():.2%}  "
          f"min={selectivities.min():.2%}  "
          f"max={selectivities.max():.2%}")

    # ------------------------------------------------------------------
    # 3. Method A: Pre-filtering (exact KNN) → also serves as ground truth
    # ------------------------------------------------------------------
    print("\n[Ground Truth] Pre-filter + exact KNN …")
    pre = PreFilterSearch(base_vecs, labels)
    gt_results, time_pre = pre.batch_search(query_vecs, filter_ranges, k=args.k)
    print(f"    Done in {time_pre:.3f}s  "
          f"({len(gt_results) / time_pre:.1f} QPS)")

    # ------------------------------------------------------------------
    # 4. Method B: Selected search method
    # ------------------------------------------------------------------
    print(f"\n[Method] {args.method.upper()} …")
    
    filter_aug_params = {
        "is_filter_augmented": args.filter_augmented,
        "alpha": args.alpha,
        "label_dim_ratio": args.label_dim_ratio,
        "n_labels": args.n_labels,
    }
    
    if args.method == "postfilter":
        search_method = PostFilterSearch(
            base_vecs,
            labels,
            k_multiplier = args.k_multiplier,
            n_tables     = args.lsh_tables,
            n_functions  = args.lsh_functions,
            bin_width    = args.lsh_bin_width,
            seed         = args.seed,
            **filter_aug_params,
        )
    elif args.method == "adaptive":
        search_method = AdaptiveFilteredSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            tau_small=args.tau_small,
            tau_medium=args.tau_medium,
            n_tables=args.lsh_tables,
            n_functions=args.lsh_functions,
            bin_width=args.lsh_bin_width,
            seed=args.seed,
            **filter_aug_params,
        )
    elif args.method == "label-sorted":
        # Label-sorted: just use PreFilterSearch for now (exact baseline)
        search_method = PreFilterSearch(base_vecs, labels)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    results, search_time = search_method.batch_search(query_vecs, filter_ranges, k=args.k)
    print(f"    Done in {search_time:.3f}s  "
          f"({len(results) / search_time:.1f} QPS)")

    # Candidate-set diagnostics
    if hasattr(search_method, 'candidate_stats'):
        cand_stats = search_method.candidate_stats(
            query_vecs[:min(200, Q)], filter_ranges[:min(200, Q)])
        print(f"\n[Candidate stats (first 200 queries)]")
        for k_s, v_s in cand_stats.items():
            print(f"    {k_s:<38} {v_s:.2f}" if isinstance(v_s, float)
                  else f"    {k_s:<38} {v_s}")

    # ------------------------------------------------------------------
    # 5. Evaluate and log
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    
    rec = compute_recall(results, gt_results)
    qps = compute_qps(Q, search_time)
    final_score = (qps / 100) * (rec["mean_recall"] ** 2)
    
    print(f"\nMean Recall@{args.k}:  {rec['mean_recall']:.4f}")
    print(f"Median Recall@{args.k}: {rec['median_recall']:.4f}")
    print(f"Min Recall@{args.k}:    {rec['min_recall']:.4f}")
    print(f"QPS:                  {qps:.1f}")
    print(f"Final Score:          {final_score:.4f}")
    
    # Log to CSV if requested
    if args.experiment_log:
        logger = ExperimentLogger(args.experiment_log)
        
        # Gather metrics
        metrics = {
            'search_time_s': search_time,
            'qps': qps,
            'mean_recall': rec['mean_recall'],
            'median_recall': rec['median_recall'],
            'min_recall': rec['min_recall'],
            'max_recall': rec['max_recall'],
            'final_score': final_score,
        }
        
        # Add candidate stats if available
        if hasattr(search_method, 'candidate_stats'):
            cand_stats = search_method.candidate_stats(query_vecs, filter_ranges)
            for k_s, v_s in cand_stats.items():
                metrics[k_s] = v_s
        
        # Gather hyperparameters
        hyperparams = {
            'alpha': args.alpha,
            'label_dim_ratio': args.label_dim_ratio,
            'n_tables': args.lsh_tables,
            'n_functions': args.lsh_functions,
            'bin_width': args.lsh_bin_width,
            'tau_small': args.tau_small if args.method == 'adaptive' else '',
            'tau_medium': args.tau_medium if args.method == 'adaptive' else '',
            'probe_radius': args.probe_radius,
        }
        
        # Dataset config
        dataset_config = {
            'seed': args.seed,
            'dataset_mode': 'sift' if args.sift else 'synthetic',
            'n_base': N,
            'n_query': Q,
            'k': args.k,
            'n_labels': args.n_labels,
            'min_sel': args.min_sel,
            'max_sel': args.max_sel,
            'mean_sel': selectivities.mean(),
        }
        
        logger.log_experiment(
            method=args.method,
            metrics=metrics,
            hyperparams=hyperparams,
            dataset_config=dataset_config,
            notes=f"Method={args.method}",
        )
        print(f"\n✓ Logged to {args.experiment_log}")

    # Print comparison table
    print_comparison(
        name_a        = "PreFilter(GT)",
        results_a     = gt_results,
        time_a        = time_pre,
        name_b        = f"{args.method.upper()}",
        results_b     = results,
        time_b        = search_time,
        groundtruth   = gt_results,
        k             = args.k,
        filter_ranges = filter_ranges,
        n_base        = N,
        n_labels      = args.n_labels,
    )


    '''
    # ------------------------------------------------------------------
    # 6. Ablation: vary LSH bin width, #tables and report recall + QPS
    # ------------------------------------------------------------------
    print("\n--- Ablation: LSH bin_width sweep (post-filter recall vs speed) ---")
    print(f"  {'lsh_tables':>10}  {'bin_width':>10} {'mean_recall':>13} {'QPS':>10}")
    print("-" * 40)

    from evaluate import compute_recall, compute_qps

    # For L2-normalised vectors distances live in [0, 2]; choose bin widths
    # that span a meaningful fraction of that range.
    for nb in [50, 100, 200, 300, 400]:
        for bw in [0.1, 0.5, 0.8, 1, 1.5]:
            p_abl = PostFilterSearch(
                base_vecs, labels,
                k_multiplier = args.k_multiplier,
                n_tables     = nb,
                n_functions  = args.lsh_functions,
                bin_width    = bw,
                seed         = args.seed,
            )
            res_abl, t_abl = p_abl.batch_search(query_vecs, filter_ranges, k=args.k)
            rec_abl = compute_recall(res_abl, gt_results)
            qps_abl = compute_qps(Q, t_abl)
            print(f"  {nb:>10d}  {bw:>10.1f} {rec_abl['mean_recall']:>13.4f} {qps_abl:>10.1f}")

    print()
    '''


if __name__ == "__main__":
    main()
