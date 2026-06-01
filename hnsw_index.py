"""
hnsw_index.py
=============
Thin wrapper around hnswlib for Euclidean ANN candidate generation.

The index is built before query timing starts. Query methods return candidate
ids only; filtering and exact L2 reranking stay in the search layer so Recall@K
and final score semantics remain unchanged.
"""

from __future__ import annotations

import numpy as np

try:
    import hnswlib
except ImportError as exc:
    hnswlib = None
    HNSWLIB_IMPORT_ERROR = exc
else:
    HNSWLIB_IMPORT_ERROR = None


class HNSWIndex:
    """
    HNSW candidate generator for L2 vector search.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    m : int
        HNSW graph out-degree parameter. Larger values usually improve recall
        at higher memory/build cost.
    ef_construction : int
        Build-time graph exploration parameter.
    ef_search : int
        Query-time graph exploration parameter.
    candidate_budget : int
        Number of nearest ids to request from HNSW before label filtering.
    seed : int
        Fixed random seed for reproducible graph construction.
    num_threads : int
        hnswlib worker threads. -1 lets hnswlib use all available threads.
    """

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 200,
        candidate_budget: int = 1000,
        seed: int = 42,
        num_threads: int = -1,
    ):
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for --method hnsw and --method adaptive-hnsw. "
                "Install it with: pip install hnswlib"
            ) from HNSWLIB_IMPORT_ERROR

        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.candidate_budget = candidate_budget
        self.seed = seed
        self.num_threads = num_threads

        self.index = hnswlib.Index(space="l2", dim=dim)
        self.n_items = 0
        self._built = False

    def build(self, base_vecs: np.ndarray) -> None:
        """Build the HNSW graph over all base vectors."""
        base_vecs = np.ascontiguousarray(base_vecs, dtype=np.float32)
        self.n_items = len(base_vecs)
        if self.n_items == 0:
            raise ValueError("Cannot build HNSWIndex with zero vectors")

        self.index.init_index(
            max_elements=self.n_items,
            ef_construction=self.ef_construction,
            M=self.m,
            random_seed=self.seed,
        )
        ids = np.arange(self.n_items, dtype=np.int32)
        self.index.add_items(base_vecs, ids, num_threads=self.num_threads)
        self.index.set_ef(max(self.ef_search, min(self.candidate_budget, self.n_items)))
        self._built = True

    def query(self, q_vec: np.ndarray, candidate_budget: int | None = None) -> np.ndarray:
        """Return candidate ids for a single query vector."""
        return self.batch_query(q_vec[np.newaxis, :], candidate_budget=candidate_budget)[0]

    def batch_query(
        self,
        query_vecs: np.ndarray,
        candidate_budget: int | None = None,
        ef_search: int | None = None,
    ) -> list[np.ndarray]:
        """Return one candidate-id array per query."""
        if not self._built:
            raise RuntimeError("HNSWIndex.build() must be called before querying")

        k = self._effective_k(candidate_budget)
        if ef_search is not None:
            self.set_ef_search(ef_search, candidate_budget=k)
        query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
        labels, _ = self.index.knn_query(
            query_vecs,
            k=k,
            num_threads=self.num_threads,
        )
        return [row.astype(np.int32, copy=False) for row in labels]

    def set_ef_search(
        self,
        ef_search: int,
        candidate_budget: int | None = None,
    ) -> None:
        """Update query-time ef without rebuilding the graph."""
        self.ef_search = int(ef_search)
        ef = max(self.ef_search, self._effective_k(candidate_budget))
        self.index.set_ef(ef)

    def _effective_k(self, candidate_budget: int | None) -> int:
        budget = self.candidate_budget if candidate_budget is None else candidate_budget
        return max(1, min(int(budget), self.n_items))
