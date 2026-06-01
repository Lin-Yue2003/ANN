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
        all_candidates = self._candidate_arrays(query_vecs, filter_ranges)
        return _candidate_stats_from_arrays(all_candidates, self.labels, filter_ranges)

    def _candidate_arrays(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        return self.hnsw.batch_query(query_vecs)


class HNSWDynamicBudgetSearch(HNSWPostFilterSearch):
    """
    HNSW post-filtering with per-query candidate budget expansion.

    Starts with a small candidate pool and expands only when the label filter
    leaves too few survivors. This ablates whether dynamic budgets recover
    recall more cheaply than always querying with a large fixed budget.
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
        initial_candidate_budget: int = 200,
        budget_expansion_factor: float = 2.0,
        min_survivors_multiplier: float = 2.0,
        seed: int = 42,
    ):
        self.initial_candidate_budget = initial_candidate_budget
        self.budget_expansion_factor = budget_expansion_factor
        self.min_survivors_multiplier = min_survivors_multiplier
        super().__init__(
            base_vecs=base_vecs,
            labels=labels,
            n_labels=n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=seed,
        )

    def search(self, query: np.ndarray, lo: int, hi: int, k: int = 10) -> np.ndarray:
        candidates = self._dynamic_candidates(query, lo, hi, k)
        return self._filter_and_rerank(query, candidates, lo, hi, k)

    def batch_search(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray,
        k: int = 10,
    ) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        results = []
        for query, (lo, hi) in zip(query_vecs, filter_ranges):
            candidates = self._dynamic_candidates(query, int(lo), int(hi), k)
            results.append(self._filter_and_rerank(query, candidates, int(lo), int(hi), k))
        return results, time.perf_counter() - t0

    def _candidate_arrays(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        if filter_ranges is None:
            return super()._candidate_arrays(query_vecs)
        return [
            self._dynamic_candidates(query, int(lo), int(hi), k=50)
            for query, (lo, hi) in zip(query_vecs, filter_ranges)
        ]

    def _dynamic_candidates(self, query: np.ndarray, lo: int, hi: int, k: int) -> np.ndarray:
        budget = max(1, min(self.initial_candidate_budget, self.candidate_budget))
        target_survivors = max(k, int(np.ceil(self.min_survivors_multiplier * k)))
        candidates = np.array([], dtype=np.int32)

        while True:
            candidates = self.hnsw.query(query, candidate_budget=budget)
            labels = self.labels[candidates]
            survivors = int(((labels >= lo) & (labels <= hi)).sum())
            if survivors >= target_survivors or budget >= self.candidate_budget:
                return candidates
            next_budget = int(np.ceil(budget * self.budget_expansion_factor))
            budget = min(self.candidate_budget, max(budget + 1, next_budget))


class HNSWFilterAugmentedSearch(HNSWPostFilterSearch):
    """
    HNSW over filter-augmented vectors, with exact reranking on originals.
    """

    def __init__(
        self,
        base_vecs: np.ndarray,
        labels: np.ndarray,
        n_labels: int,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200,
        candidate_budget: int = 1000,
        hnsw_alpha: float = 0.6,
        hnsw_label_dim_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.base_vecs = np.ascontiguousarray(base_vecs, dtype=np.float32)
        self.labels = labels.astype(np.int32, copy=False)
        self.n_labels = n_labels
        self.N, self.D = self.base_vecs.shape
        self.candidate_budget = candidate_budget
        self.hnsw_alpha = hnsw_alpha
        self.hnsw_label_dim_ratio = hnsw_label_dim_ratio
        self.label_dim = max(1, int(round(self.D * hnsw_label_dim_ratio)))
        self._sqrt_alpha = float(np.sqrt(hnsw_alpha))
        self._sqrt_1_alpha = float(np.sqrt(max(0.0, 1.0 - hnsw_alpha)))

        aug_dim = self.D + self.label_dim
        self.hnsw = HNSWIndex(
            dim=aug_dim,
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=seed,
        )

        print(
            f"[HNSWFilterAug] Building augmented graph over {self.N:,} vectors "
            f"(alpha={hnsw_alpha}, label_dim={self.label_dim}) ..."
        )
        t0 = time.perf_counter()
        self.hnsw.build(self._augment_base(self.base_vecs, self.labels))
        print(f"[HNSWFilterAug] Index built in {time.perf_counter() - t0:.2f}s")

    def search(self, query: np.ndarray, lo: int, hi: int, k: int = 10) -> np.ndarray:
        candidates = self._candidate_arrays(query[np.newaxis, :], np.array([[lo, hi]]))[0]
        return self._filter_and_rerank(query, candidates, lo, hi, k)

    def batch_search(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray,
        k: int = 10,
    ) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        all_candidates = self._candidate_arrays(query_vecs, filter_ranges)
        results = []
        for query, candidates, (lo, hi) in zip(query_vecs, all_candidates, filter_ranges):
            results.append(self._filter_and_rerank(query, candidates, int(lo), int(hi), k))
        return results, time.perf_counter() - t0

    def _candidate_arrays(
        self,
        query_vecs: np.ndarray,
        filter_ranges: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        if filter_ranges is None:
            raise ValueError("filter_ranges are required for HNSWFilterAugmentedSearch")
        return self.hnsw.batch_query(self._augment_query(query_vecs, filter_ranges))

    def _augment_base(self, vecs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        label_values = labels.astype(np.float32) / max(1, self.n_labels)
        label_vecs = np.broadcast_to(label_values[:, None], (len(vecs), self.label_dim))
        return np.column_stack([
            self._sqrt_alpha * vecs,
            self._sqrt_1_alpha * label_vecs,
        ]).astype(np.float32, copy=False)

    def _augment_query(self, query_vecs: np.ndarray, filter_ranges: np.ndarray) -> np.ndarray:
        midpoints = (filter_ranges[:, 0] + filter_ranges[:, 1]).astype(np.float32) / 2.0
        label_values = midpoints / max(1, self.n_labels)
        label_vecs = np.broadcast_to(label_values[:, None], (len(query_vecs), self.label_dim))
        return np.column_stack([
            self._sqrt_alpha * query_vecs,
            self._sqrt_1_alpha * label_vecs,
        ]).astype(np.float32, copy=False)


def _candidate_stats_from_arrays(
    all_candidates: list[np.ndarray],
    labels: np.ndarray,
    filter_ranges: np.ndarray,
) -> dict:
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


class AdaptiveHNSWAugmentedSearch:
    """
    Selectivity-aware hybrid with separate global and filter-augmented HNSW.

    Routing:
    - small filters: exact label-sorted scan
    - medium filters: filter-augmented HNSW
    - large filters: global vector HNSW
    """

    def __init__(
        self,
        base_vecs: np.ndarray,
        labels: np.ndarray,
        n_labels: int,
        tau_small: float = 0.01,
        tau_medium: float = 0.15,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 200,
        candidate_budget: int = 1000,
        hnsw_alpha: float = 0.6,
        hnsw_label_dim_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.base_vecs = base_vecs
        self.labels = labels
        self.n_labels = n_labels
        self.tau_small = tau_small
        self.tau_medium = tau_medium

        print(
            "[AdaptiveHNSWAug] Initializing "
            f"tau_small={tau_small}, tau_medium={tau_medium}, "
            f"alpha={hnsw_alpha}, label_dim_ratio={hnsw_label_dim_ratio}"
        )

        self.exact_small = LabelSortedPreFilterSearch(base_vecs, labels, n_labels)
        self.global_hnsw = HNSWPostFilterSearch(
            base_vecs,
            labels,
            n_labels=n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            seed=seed,
        )
        self.aug_hnsw = HNSWFilterAugmentedSearch(
            base_vecs,
            labels,
            n_labels=n_labels,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            candidate_budget=candidate_budget,
            hnsw_alpha=hnsw_alpha,
            hnsw_label_dim_ratio=hnsw_label_dim_ratio,
            seed=seed,
        )

    def search(self, query: np.ndarray, lo: int, hi: int, k: int = 10) -> np.ndarray:
        selectivity = (hi - lo + 1) / self.n_labels
        if selectivity <= self.tau_small:
            return self.exact_small.search(query, lo, hi, k)
        if selectivity <= self.tau_medium:
            return self.aug_hnsw.search(query, lo, hi, k)
        return self.global_hnsw.search(query, lo, hi, k)

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
            print(f"[AdaptiveHNSWAug] {len(small_indices)} small-selectivity queries -> exact")
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
            print(f"[AdaptiveHNSWAug] {len(medium_indices)} medium-selectivity queries -> augmented hnsw")
            medium_results, medium_time = self.aug_hnsw.batch_search(
                query_vecs[medium_indices],
                filter_ranges[medium_indices],
                k,
            )
            for i, idx in enumerate(medium_indices):
                results[idx] = medium_results[i]
            total_time += medium_time

        large_indices = np.where(large_mask)[0]
        if len(large_indices) > 0:
            print(f"[AdaptiveHNSWAug] {len(large_indices)} large-selectivity queries -> global hnsw")
            large_results, large_time = self.global_hnsw.batch_search(
                query_vecs[large_indices],
                filter_ranges[large_indices],
                k,
            )
            for i, idx in enumerate(large_indices):
                results[idx] = large_results[i]
            total_time += large_time

        return [r if r is not None else np.array([], dtype=np.int32) for r in results], total_time

    def candidate_stats(self, query_vecs: np.ndarray, filter_ranges: np.ndarray) -> dict:
        lo_vals = filter_ranges[:, 0]
        hi_vals = filter_ranges[:, 1]
        selectivities = (hi_vals - lo_vals + 1) / self.n_labels
        all_candidates: list[np.ndarray] = [np.array([], dtype=np.int32) for _ in range(len(query_vecs))]

        small_indices = np.where(selectivities <= self.tau_small)[0]
        for idx in small_indices:
            lo, hi = filter_ranges[idx]
            all_candidates[idx] = self.exact_small._range_ids(int(lo), int(hi))

        medium_mask = (selectivities > self.tau_small) & (selectivities <= self.tau_medium)
        medium_indices = np.where(medium_mask)[0]
        if len(medium_indices) > 0:
            medium_candidates = self.aug_hnsw._candidate_arrays(
                query_vecs[medium_indices],
                filter_ranges[medium_indices],
            )
            for i, idx in enumerate(medium_indices):
                all_candidates[idx] = medium_candidates[i]

        large_indices = np.where(selectivities > self.tau_medium)[0]
        if len(large_indices) > 0:
            large_candidates = self.global_hnsw._candidate_arrays(
                query_vecs[large_indices],
                filter_ranges[large_indices],
            )
            for i, idx in enumerate(large_indices):
                all_candidates[idx] = large_candidates[i]

        return _candidate_stats_from_arrays(all_candidates, self.labels, filter_ranges)
