"""
hnsw_ablation_v2_runner.py
==========================
Focused HNSW-filter-aug search with graph reuse and multi-seed validation.

Reuse strategy:
- Load SIFT/base/query vectors once.
- For each seed, generate labels/filter ranges and exact ground truth once.
- For each seed and (hnsw_alpha, hnsw_label_dim_ratio), build one augmented
  HNSW graph, then reuse it across ef_search and candidate_budget values.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import assign_labels, generate_filter_ranges, generate_synthetic_data, load_sift
from evaluate import compute_qps, compute_recall
from experiments.logger import ExperimentLogger
from hnsw_index import HNSWLIB_IMPORT_ERROR
from hnsw_search import HNSWFilterAugmentedSearch
from prefilter import PreFilterSearch
from sweep_params import generate_hnsw_ablation_v2_configs


class ReusableFilterAugmentedHNSW(HNSWFilterAugmentedSearch):
    """Filter-augmented HNSW that allows ef/budget changes without rebuild."""

    def batch_search_with_params(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray,
        k: int,
        ef_search: int,
        candidate_budget: int,
    ) -> tuple[list[np.ndarray], float, list[np.ndarray]]:
        t0 = time.perf_counter()
        augmented_queries = self._augment_query(query_vecs, filter_ranges)
        all_candidates = self.hnsw.batch_query(
            augmented_queries,
            candidate_budget=candidate_budget,
            ef_search=ef_search,
        )

        results = []
        for query, candidates, (lo, hi) in zip(query_vecs, all_candidates, filter_ranges):
            results.append(self._filter_and_rerank(query, candidates, int(lo), int(hi), k))
        return results, time.perf_counter() - t0, all_candidates


def parse_args():
    p = argparse.ArgumentParser(
        description="Focused HNSW-filter-aug v2 ablation with graph reuse"
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
    p.add_argument("--seeds", default="40,41,42",
                   help="Comma-separated seeds for labels, ranges, and HNSW construction")
    p.add_argument("--hnsw-m", type=int, default=16)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--experiment-log", default="experiments/hnsw_ablation_v2_results.csv")
    p.add_argument("--limit-configs", type=int, default=None,
                   help="Run only the first N v2 configs per seed")
    p.add_argument("--candidate-stats-limit", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    if HNSWLIB_IMPORT_ERROR is not None:
        print("[error] hnswlib is required. Run: uv sync")
        raise SystemExit(1)

    print("\n" + "=" * 76)
    print("  HNSW Ablation V2: focused filter-aug search")
    print("=" * 76)

    base_vecs, query_vecs = load_data(args)
    n_base, dim = base_vecs.shape
    n_query = len(query_vecs)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    configs = generate_hnsw_ablation_v2_configs()
    if args.limit_configs is not None:
        configs = configs[:args.limit_configs]

    print(f"[config] N={n_base:,} Q={n_query:,} D={dim} K={args.k}")
    print(f"[config] seeds={seeds}")
    print(f"[config] configs per seed={len(configs)}")
    print(f"[config] graph groups per seed={count_graph_groups(configs)}")

    logger = ExperimentLogger(args.experiment_log)

    for seed_idx, seed in enumerate(seeds, start=1):
        print("\n" + "#" * 76)
        print(f"[Seed {seed_idx}/{len(seeds)}] seed={seed}")
        print("#" * 76)

        labels = assign_labels(n_base, n_labels=args.n_labels, seed=seed)
        filter_ranges = generate_filter_ranges(
            n_query,
            n_labels=args.n_labels,
            min_selectivity=args.min_sel,
            max_selectivity=args.max_sel,
            seed=seed + 1,
        )
        selectivities = (filter_ranges[:, 1] - filter_ranges[:, 0] + 1) / args.n_labels

        print(
            f"[Seed {seed}] selectivity mean={selectivities.mean():.2%} "
            f"min={selectivities.min():.2%} max={selectivities.max():.2%}"
        )
        print(f"[Seed {seed}] Building exact ground truth once ...")
        gt_t0 = time.perf_counter()
        gt_results, gt_time = PreFilterSearch(base_vecs, labels).batch_search(
            query_vecs,
            filter_ranges,
            k=args.k,
        )
        print(f"[Seed {seed}] Ground truth done in {gt_time:.3f}s")

        grouped = group_configs_by_graph(configs)
        config_counter = 0
        for graph_key, graph_configs in grouped.items():
            hnsw_alpha, label_dim_ratio = graph_key
            print("\n" + "-" * 76)
            print(
                f"[Seed {seed}] Build graph alpha={hnsw_alpha} "
                f"label_dim_ratio={label_dim_ratio} configs={len(graph_configs)}"
            )
            graph_t0 = time.perf_counter()
            searcher = ReusableFilterAugmentedHNSW(
                base_vecs,
                labels,
                n_labels=args.n_labels,
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                hnsw_ef_search=max(int(c["hnsw_ef_search"]) for c in graph_configs),
                candidate_budget=max(int(c["candidate_budget"]) for c in graph_configs),
                hnsw_alpha=float(hnsw_alpha),
                hnsw_label_dim_ratio=float(label_dim_ratio),
                seed=seed,
            )
            graph_build_time = time.perf_counter() - graph_t0
            print(f"[Seed {seed}] Graph built in {graph_build_time:.3f}s")

            for cfg in graph_configs:
                config_counter += 1
                ef_search = int(cfg["hnsw_ef_search"])
                candidate_budget = int(cfg["candidate_budget"])
                print(
                    f"[Seed {seed}] Config {config_counter}/{len(configs)} "
                    f"ef={ef_search} budget={candidate_budget} "
                    f"alpha={hnsw_alpha} label_dim={label_dim_ratio}"
                )

                results, search_time, all_candidates = searcher.batch_search_with_params(
                    query_vecs,
                    filter_ranges,
                    k=args.k,
                    ef_search=ef_search,
                    candidate_budget=candidate_budget,
                )
                rec = compute_recall(results, gt_results)
                qps = compute_qps(n_query, search_time)
                final_score = (qps / 100) * (rec["mean_recall"] ** 2)

                print(
                    f"[Seed {seed}] score={final_score:.4f} "
                    f"recall={rec['mean_recall']:.4f} qps={qps:.1f} "
                    f"search_time={search_time:.3f}s"
                )

                n_stats = min(args.candidate_stats_limit, n_query)
                metrics = {
                    "search_time_s": search_time,
                    "qps": qps,
                    "mean_recall": rec["mean_recall"],
                    "median_recall": rec["median_recall"],
                    "min_recall": rec["min_recall"],
                    "max_recall": rec["max_recall"],
                    "final_score": final_score,
                }
                if n_stats > 0:
                    metrics.update(candidate_stats_from_arrays(
                        all_candidates[:n_stats],
                        labels,
                        filter_ranges[:n_stats],
                    ))

                logger.log_experiment(
                    method="hnsw-filter-aug-v2",
                    metrics=metrics,
                    hyperparams={
                        "hnsw_m": args.hnsw_m,
                        "hnsw_ef_construction": args.hnsw_ef_construction,
                        "hnsw_ef_search": ef_search,
                        "candidate_budget": candidate_budget,
                        "hnsw_alpha": hnsw_alpha,
                        "hnsw_label_dim_ratio": label_dim_ratio,
                    },
                    dataset_config={
                        "seed": seed,
                        "dataset_mode": "sift" if args.sift else "synthetic",
                        "n_base": n_base,
                        "n_query": n_query,
                        "k": args.k,
                        "n_labels": args.n_labels,
                        "min_sel": args.min_sel,
                        "max_sel": args.max_sel,
                        "mean_sel": selectivities.mean(),
                    },
                    notes=(
                        f"v2 graph_reuse; gt_time_s={gt_time:.3f}; "
                        f"graph_build_time_s={graph_build_time:.3f}"
                    ),
                )

        print(f"[Seed {seed}] seed total wall time={time.perf_counter() - gt_t0:.3f}s")

    print("\n" + "=" * 76)
    print(f"[V2] Done. Results: {args.experiment_log}")
    print("=" * 76)


def load_data(args):
    if args.sift:
        return load_sift(
            base_dir=args.sift_dir,
            max_base=args.max_base,
            max_query=args.n_query,
        )
    return generate_synthetic_data(
        n_base=args.max_base,
        n_query=args.n_query,
        dim=args.dim,
        seed=42,
    )


def group_configs_by_graph(configs):
    grouped = defaultdict(list)
    for cfg in configs:
        key = (float(cfg["hnsw_alpha"]), float(cfg["hnsw_label_dim_ratio"]))
        grouped[key].append(cfg)
    return dict(sorted(grouped.items()))


def count_graph_groups(configs):
    return len(group_configs_by_graph(configs))


def candidate_stats_from_arrays(all_candidates, labels, filter_ranges):
    total, surviving = [], []
    for candidates, (lo, hi) in zip(all_candidates, filter_ranges):
        total.append(len(candidates))
        if len(candidates):
            mask = (labels[candidates] >= lo) & (labels[candidates] <= hi)
            surviving.append(int(mask.sum()))
        else:
            surviving.append(0)
    tc = np.array(total, dtype=np.int32)
    sc = np.array(surviving, dtype=np.int32)
    return {
        "avg_total_candidates": float(tc.mean()),
        "avg_surviving_candidates": float(sc.mean()),
        "avg_survival_rate": float((sc / np.maximum(tc, 1)).mean()),
        "queries_with_zero_candidates": int((tc == 0).sum()),
        "queries_with_zero_surviving": int((sc == 0).sum()),
    }


if __name__ == "__main__":
    main()
