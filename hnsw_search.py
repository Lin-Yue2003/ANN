"""
hnsw_search.py
==============
Filtered search methods backed by a global HNSW graph index.
"""

from __future__ import annotations

import time

import numpy as np

from hnsw_index import HNSWIndex
from postfilter import PostFilterSearch


class LabelSortedPreFilterSearch:
    """
    Exact filtered KNN using a pre-sorted label index.

    This keeps exact ground-truth semantics for small filter ranges while
    avoiding an O(N) boolean mask scan for every query.
    """

    def __init__(self, base_vecs: np.ndarray, labels: np.ndarray, n_labels: int):
        self.base_vecs = base_vecs
        self.labels = labels.astype(np.int32, copy=False)
        self.n_labels = n_labels
        self.N, self.D = base_vecs.shape

        self.sort_order = np.argsort(self.labels, kind="stable").astype(np.int32)
        sorted_labels = self.labels[self.sort_order]
        boundaries = np.searchsorted(sorted_labels, np.arange(n_labels + 1), side="left")
        self.label_start = boundaries[:-1].astype(np.int32)
        self.label_end_exclusive = boundaries[1:].astype(np.int32)

    def search(self, query: np.ndarray, lo: int, hi: int, k: int = 10) -> np.ndarray:
        valid = self._range_ids(lo, hi)
        if len(valid) == 0:
            return np.array([], dtype=np.int32)
        return self._rerank(query, valid, k)

    def batch_search(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray,
        k: int = 10,
    ) -> tuple[list[np.ndarray], float]:
        results = []
        t0 = time.perf_counter()
        for query, (lo, hi) in zip(query_vecs, filter_ranges):
            results.append(self.search(query, int(lo), int(hi), k))
        return results, time.perf_counter() - t0

    def _range_ids(self, lo: int, hi: int) -> np.ndarray:
        lo = max(0, min(int(lo), self.n_labels - 1))
        hi = max(0, min(int(hi), self.n_labels - 1))
        if lo > hi:
            return np.array([], dtype=np.int32)

        start = int(self.label_start[lo])
        stop = int(self.label_end_exclusive[hi])
        if start >= stop:
            return np.array([], dtype=np.int32)
        return self.sort_order[start:stop]

    def _rerank(self, query: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
        diff = self.base_vecs[valid] - query
        dists = np.einsum("nd,nd->n", diff, diff)
        k_eff = min(k, len(valid))
        top_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return valid[top_idx]


class HNSWPostFilterSearch:
    """
    HNSW candidate generation followed by exact label filter and L2 rerank.
    """

    def __init__(
        self,
        base_vecs: np.ndarray,
        labels: np.ndarray,
        n_labels: int | None = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200,
        candidate_budget: int = 1000,
        seed: int = 42,
    ):
        self.base_vecs = np.ascontiguousarray(base_vecs, dtype=np.float32)
        self.labels = labels.astype(np.int32, copy=False)
        self.n_labels = n_labels
        self.N, self.D = self.base_vecs.shape
        self.candidate_budget = candidate_budget

        self.hnsw = HNSWIndex(
            dim=self.D,
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=seed,
        )

        print(f"[HNSW] Building graph over {self.N:,} vectors ...")
        t0 = time.perf_counter()
        self.hnsw.build(self.base_vecs)
        print(f"[HNSW] Index built in {time.perf_counter() - t0:.2f}s")

    def search(self, query: np.ndarray, lo: int, hi: int, k: int = 10) -> np.ndarray:
        candidates = self.hnsw.query(query)
        return self._filter_and_rerank(query, candidates, lo, hi, k)

    def batch_search(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray,
        k: int = 10,
    ) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        all_candidates = self.hnsw.batch_query(query_vecs)

        results = []
        for query, candidates, (lo, hi) in zip(query_vecs, all_candidates, filter_ranges):
            results.append(self._filter_and_rerank(query, candidates, int(lo), int(hi), k))
        return results, time.perf_counter() - t0

    def _filter_and_rerank(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        lo: int,
        hi: int,
        k: int,
    ) -> np.ndarray:
        if len(candidates) == 0:
            return np.array([], dtype=np.int32)

        lbl = self.labels[candidates]
        valid = candidates[(lbl >= lo) & (lbl <= hi)]
        if len(valid) == 0:
            return np.array([], dtype=np.int32)

        diff = self.base_vecs[valid] - query
        dists = np.einsum("nd,nd->n", diff, diff)
        k_eff = min(k, len(valid))
        top_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        return valid[top_idx]

    def candidate_stats(self, query_vecs: np.ndarray, filter_ranges: np.ndarray) -> dict:
        all_candidates = self.hnsw.batch_query(query_vecs)
        total, surviving = [], []
        for candidates, (lo, hi) in zip(all_candidates, filter_ranges):
            total.append(len(candidates))
            if len(candidates):
                mask = (self.labels[candidates] >= lo) & (self.labels[candidates] <= hi)
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


class AdaptiveHNSWSearch:
    """
    Selectivity-aware hybrid using label-sorted exact search and HNSW.
    """

    def __init__(
        self,
        base_vecs: np.ndarray,
        labels: np.ndarray,
        n_labels: int,
        tau_small: float = 0.01,
        tau_medium: float = 0.15,
        adaptive_medium_index: str = "hnsw",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200,
        candidate_budget: int = 1000,
        **lsh_params,
    ):
        self.base_vecs = base_vecs
        self.labels = labels
        self.n_labels = n_labels
        self.tau_small = tau_small
        self.tau_medium = tau_medium
        self.adaptive_medium_index = adaptive_medium_index

        print(
            "[AdaptiveHNSW] Initializing "
            f"tau_small={tau_small}, tau_medium={tau_medium}, "
            f"medium={adaptive_medium_index}"
        )

        self.exact_small = LabelSortedPreFilterSearch(base_vecs, labels, n_labels)
        self.hnsw = HNSWPostFilterSearch(
            base_vecs,
            labels,
            n_labels=n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=lsh_params.get("seed", 42),
        )

        self.lsh_medium = None
        if adaptive_medium_index == "lsh":
            lsh_medium_params = lsh_params.copy()
            lsh_medium_params["is_filter_augmented"] = True
            lsh_medium_params["n_labels"] = n_labels
            self.lsh_medium = PostFilterSearch(base_vecs, labels, **lsh_medium_params)
        elif adaptive_medium_index != "hnsw":
            raise ValueError("adaptive_medium_index must be 'hnsw' or 'lsh'")

    def search(self, query: np.ndarray, lo: int, hi: int, k: int = 10) -> np.ndarray:
        selectivity = (hi - lo + 1) / self.n_labels
        if selectivity <= self.tau_small:
            return self.exact_small.search(query, lo, hi, k)
        if selectivity <= self.tau_medium and self.lsh_medium is not None:
            return self.lsh_medium.search(query, lo, hi, k)
        return self.hnsw.search(query, lo, hi, k)

    def batch_search(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray,
        k: int = 10,
    ) -> tuple[list[np.ndarray], float]:
        lo_vals = filter_ranges[:, 0]
        hi_vals = filter_ranges[:, 1]
        selectivities = (hi_vals - lo_vals + 1) / self.n_labels

        small_mask = selectivities <= self.tau_small
        medium_mask = (selectivities > self.tau_small) & (selectivities <= self.tau_medium)
        large_mask = selectivities > self.tau_medium

        results: list[np.ndarray | None] = [None] * len(query_vecs)
        total_time = 0.0

        small_indices = np.where(small_mask)[0]
        if len(small_indices) > 0:
            print(f"[AdaptiveHNSW] {len(small_indices)} small-selectivity queries -> exact")
            small_results, small_time = self.exact_small.batch_search(
                query_vecs[small_indices],
                filter_ranges[small_indices],
                k,
            )
            for i, idx in enumerate(small_indices):
                results[idx] = small_results[i]
            total_time += small_time

        medium_indices = np.where(medium_mask)[0]
        if len(medium_indices) > 0:
            medium_search = self.lsh_medium if self.lsh_medium is not None else self.hnsw
            print(
                f"[AdaptiveHNSW] {len(medium_indices)} medium-selectivity queries "
                f"-> {self.adaptive_medium_index}"
            )
            medium_results, medium_time = medium_search.batch_search(
                query_vecs[medium_indices],
                filter_ranges[medium_indices],
                k,
            )
            for i, idx in enumerate(medium_indices):
                results[idx] = medium_results[i]
            total_time += medium_time

        large_indices = np.where(large_mask)[0]
        if len(large_indices) > 0:
            print(f"[AdaptiveHNSW] {len(large_indices)} large-selectivity queries -> hnsw")
            large_results, large_time = self.hnsw.batch_search(
                query_vecs[large_indices],
                filter_ranges[large_indices],
                k,
            )
            for i, idx in enumerate(large_indices):
                results[idx] = large_results[i]
            total_time += large_time

        return [r if r is not None else np.array([], dtype=np.int32) for r in results], total_time

    def candidate_stats(self, query_vecs: np.ndarray, filter_ranges: np.ndarray) -> dict:
        return self.hnsw.candidate_stats(query_vecs, filter_ranges)
