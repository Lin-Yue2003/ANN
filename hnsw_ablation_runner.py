"""
hnsw_ablation_runner.py
=======================
Batch HNSW ablation runner that reuses dataset and ground truth.

Unlike repeatedly launching ``main.py``, this loads/generates data once,
computes the exact prefilter ground truth once, then runs all HNSW ablation
configs in the same process and appends each result to one CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from adaptive_search import AdaptiveFilteredSearch
from data_utils import assign_labels, generate_filter_ranges, generate_synthetic_data, load_sift
from evaluate import compute_qps, compute_recall
from experiments.logger import ExperimentLogger
from hnsw_index import HNSWLIB_IMPORT_ERROR
from hnsw_search import (
    AdaptiveHNSWAugmentedSearch,
    AdaptiveHNSWSearch,
    HNSWDynamicBudgetSearch,
    HNSWFilterAugmentedSearch,
    HNSWPostFilterSearch,
    LabelSortedPreFilterSearch,
)
from postfilter import PostFilterSearch
from prefilter import PreFilterSearch
from sweep_params import generate_hnsw_ablation_configs


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch HNSW ablation runner with shared ground truth"
    )
    p.add_argument("--sift", action="store_true", help="Load SIFT instead of synthetic data")
    p.add_argument("--sift-dir", default="./data")
    p.add_argument("--max-base", type=int, default=None)
    p.add_argument("--n-query", type=int, default=None)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-labels", type=int, default=1000)
    p.add_argument("--min-sel", type=float, default=0.05)
    p.add_argument("--max-sel", type=float, default=0.40)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-log", default="experiments/hnsw_ablation_results.csv")
    p.add_argument("--limit-configs", type=int, default=None,
                   help="Run only the first N configs for sanity checks")
    p.add_argument("--candidate-stats-limit", type=int, default=200,
                   help="Number of queries used for candidate diagnostics per config")
    return p.parse_args()


def main():
    args = parse_args()
    if HNSWLIB_IMPORT_ERROR is not None:
        print(
            "[error] hnswlib is required for HNSW ablation. "
            "Run: uv sync  (or pip install -r requirements.txt)"
        )
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("  Batch HNSW Ablation")
    print("=" * 70)

    base_vecs, query_vecs = load_data(args)
    n_base, dim = base_vecs.shape
    n_query = len(query_vecs)

    labels = assign_labels(n_base, n_labels=args.n_labels, seed=args.seed)
    filter_ranges = generate_filter_ranges(
        n_query,
        n_labels=args.n_labels,
        min_selectivity=args.min_sel,
        max_selectivity=args.max_sel,
        seed=args.seed + 1,
    )
    selectivities = (filter_ranges[:, 1] - filter_ranges[:, 0] + 1) / args.n_labels

    print(
        f"[config] N={n_base:,} Q={n_query:,} D={dim} K={args.k} "
        f"n_labels={args.n_labels}"
    )
    print(
        f"[config] selectivity mean={selectivities.mean():.2%} "
        f"min={selectivities.min():.2%} max={selectivities.max():.2%}"
    )

    print("\n[Ground Truth] Pre-filter + exact KNN once ...")
    pre = PreFilterSearch(base_vecs, labels)
    gt_results, gt_time = pre.batch_search(query_vecs, filter_ranges, k=args.k)
    print(f"[Ground Truth] Done in {gt_time:.3f}s ({n_query / gt_time:.1f} QPS)")

    configs = generate_hnsw_ablation_configs()
    if args.limit_configs is not None:
        configs = configs[:args.limit_configs]

    print(f"\n[Runner] Running {len(configs)} configs")
    logger = ExperimentLogger(args.experiment_log)

    for i, cfg in enumerate(configs, start=1):
        print("\n" + "-" * 70)
        print(f"[Config {i}/{len(configs)}] {format_config(cfg)}")
        print("-" * 70)

        t_build0 = time.perf_counter()
        search_method = build_search_method(cfg, base_vecs, labels, args)
        build_time = time.perf_counter() - t_build0
        print(f"[Config {i}] Build/setup time: {build_time:.3f}s")

        results, search_time = search_method.batch_search(query_vecs, filter_ranges, k=args.k)
        rec = compute_recall(results, gt_results)
        qps = compute_qps(n_query, search_time)
        final_score = (qps / 100) * (rec["mean_recall"] ** 2)

        print(
            f"[Config {i}] recall={rec['mean_recall']:.4f} "
            f"qps={qps:.1f} score={final_score:.4f} search_time={search_time:.3f}s"
        )

        metrics = {
            "search_time_s": search_time,
            "qps": qps,
            "mean_recall": rec["mean_recall"],
            "median_recall": rec["median_recall"],
            "min_recall": rec["min_recall"],
            "max_recall": rec["max_recall"],
            "final_score": final_score,
        }

        if hasattr(search_method, "candidate_stats") and args.candidate_stats_limit > 0:
            n_stats = min(args.candidate_stats_limit, n_query)
            metrics.update(search_method.candidate_stats(
                query_vecs[:n_stats],
                filter_ranges[:n_stats],
            ))

        logger.log_experiment(
            method=cfg["method"],
            metrics=metrics,
            hyperparams=hyperparams_for_log(cfg),
            dataset_config={
                "seed": args.seed,
                "dataset_mode": "sift" if args.sift else "synthetic",
                "n_base": n_base,
                "n_query": n_query,
                "k": args.k,
                "n_labels": args.n_labels,
                "min_sel": args.min_sel,
                "max_sel": args.max_sel,
                "mean_sel": selectivities.mean(),
            },
            notes=f"Batch config {i}/{len(configs)}; gt_time_s={gt_time:.3f}",
        )

    print("\n" + "=" * 70)
    print(f"[Runner] Done. Results: {args.experiment_log}")
    print("=" * 70)


def load_data(args):
    if args.sift:
        return load_sift(
            base_dir=args.sift_dir,
            max_base=args.max_base,
            max_query=args.n_query,
        )
    base_vecs, query_vecs = generate_synthetic_data(
        n_base=args.max_base,
        n_query=args.n_query,
        dim=args.dim,
        seed=args.seed,
    )
    return base_vecs, query_vecs[:args.n_query]


def build_search_method(cfg, base_vecs, labels, args):
    method = cfg["method"]
    hnsw_m = int(cfg.get("hnsw_m", 16))
    hnsw_ef_construction = int(cfg.get("hnsw_ef_construction", 200))
    hnsw_ef_search = int(cfg.get("hnsw_ef_search", 200))
    candidate_budget = int(cfg.get("candidate_budget", 1000))

    if method == "hnsw":
        return HNSWPostFilterSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=args.seed,
        )

    if method == "hnsw-dynamic":
        return HNSWDynamicBudgetSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            initial_candidate_budget=int(cfg.get("initial_candidate_budget", 200)),
            budget_expansion_factor=float(cfg.get("budget_expansion_factor", 2.0)),
            min_survivors_multiplier=float(cfg.get("min_survivors_multiplier", 2.0)),
            seed=args.seed,
        )

    if method == "hnsw-filter-aug":
        return HNSWFilterAugmentedSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            hnsw_alpha=float(cfg.get("hnsw_alpha", 0.6)),
            hnsw_label_dim_ratio=float(cfg.get("hnsw_label_dim_ratio", 0.05)),
            seed=args.seed,
        )

    if method == "adaptive-hnsw":
        return AdaptiveHNSWSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            tau_small=float(cfg.get("tau_small", 0.01)),
            tau_medium=float(cfg.get("tau_medium", 0.15)),
            adaptive_medium_index=cfg.get("adaptive_medium_index", "hnsw"),
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=args.seed,
        )

    if method == "adaptive-hnsw-aug":
        return AdaptiveHNSWAugmentedSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            tau_small=float(cfg.get("tau_small", 0.01)),
            tau_medium=float(cfg.get("tau_medium", 0.15)),
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            hnsw_alpha=float(cfg.get("hnsw_alpha", 0.6)),
            hnsw_label_dim_ratio=float(cfg.get("hnsw_label_dim_ratio", 0.05)),
            seed=args.seed,
        )

    if method == "postfilter":
        return PostFilterSearch(
            base_vecs,
            labels,
            n_tables=int(cfg.get("n_tables", 400)),
            n_functions=int(cfg.get("n_functions", 5)),
            bin_width=float(cfg.get("bin_width", 0.22)),
            seed=args.seed,
            is_filter_augmented=True,
            alpha=float(cfg.get("alpha", 0.05)),
            label_dim_ratio=float(cfg.get("label_dim_ratio", 0.05)),
            n_labels=args.n_labels,
        )

    if method == "adaptive":
        return AdaptiveFilteredSearch(
            base_vecs,
            labels,
            n_labels=args.n_labels,
            tau_small=float(cfg.get("tau_small", 0.01)),
            tau_medium=float(cfg.get("tau_medium", 0.15)),
            n_tables=int(cfg.get("n_tables", 400)),
            n_functions=int(cfg.get("n_functions", 5)),
            bin_width=float(cfg.get("bin_width", 0.22)),
            seed=args.seed,
            is_filter_augmented=True,
            alpha=float(cfg.get("alpha", 0.05)),
            label_dim_ratio=float(cfg.get("label_dim_ratio", 0.05)),
        )

    if method == "label-sorted":
        return LabelSortedPreFilterSearch(base_vecs, labels, args.n_labels)

    raise ValueError(f"Unknown method: {method}")


def hyperparams_for_log(cfg):
    fields = [
        "alpha",
        "label_dim_ratio",
        "n_tables",
        "n_functions",
        "bin_width",
        "tau_small",
        "tau_medium",
        "adaptive_medium_index",
        "hnsw_m",
        "hnsw_ef_construction",
        "hnsw_ef_search",
        "candidate_budget",
        "initial_candidate_budget",
        "budget_expansion_factor",
        "min_survivors_multiplier",
        "hnsw_alpha",
        "hnsw_label_dim_ratio",
        "probe_radius",
    ]
    return {field: cfg.get(field, "") for field in fields}


def format_config(cfg):
    return " ".join(f"{key}={value}" for key, value in cfg.items())


if __name__ == "__main__":
    main()
