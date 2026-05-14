"""
adaptive_search.py
==================
Adaptive hybrid search by filter selectivity.

Routes queries to different search methods based on filter selectivity:
- Small selectivity: exact pre-filter scan
- Medium selectivity: filter-augmented LSH
- Large selectivity: vector-dominant LSH
"""

import time
import numpy as np
from prefilter import PreFilterSearch
from postfilter import PostFilterSearch


class AdaptiveFilteredSearch:
    """
    Adaptive search strategy based on filter selectivity.
    
    Parameters
    ----------
    base_vecs : (N, D) float32
    labels : (N,) int32
    n_labels : int
    tau_small : float
        Selectivity threshold: if sel <= tau_small, use exact search
    tau_medium : float
        Selectivity threshold: if tau_small < sel <= tau_medium, use filter-augmented LSH
        Otherwise, use vector-dominant LSH
    **lsh_params : forwarded to PostFilterSearch and PreFilterSearch
    """
    
    def __init__(self,
                 base_vecs: np.ndarray,
                 labels: np.ndarray,
                 n_labels: int,
                 tau_small: float = 0.01,
                 tau_medium: float = 0.15,
                 **lsh_params):
        
        self.base_vecs = base_vecs
        self.labels = labels
        self.n_labels = n_labels
        self.tau_small = tau_small
        self.tau_medium = tau_medium
        self.N, self.D = base_vecs.shape
        
        print(f"[Adaptive] Initializing with tau_small={tau_small}, tau_medium={tau_medium}")
        
        # Build exact search baseline (needed for small selectivity)
        self.exact_search = PreFilterSearch(base_vecs, labels)
        
        # Build LSH indices for medium and large selectivity
        # For medium: use filter-augmented LSH
        lsh_params_medium = lsh_params.copy()
        lsh_params_medium['is_filter_augmented'] = True
        self.lsh_medium = PostFilterSearch(base_vecs, labels, **lsh_params_medium)
        
        # For large: use vector-dominant LSH (low alpha to reduce filter emphasis)
        lsh_params_large = lsh_params.copy()
        lsh_params_large['is_filter_augmented'] = True
        lsh_params_large['alpha'] = 0.8  # Mostly vector-based
        self.lsh_large = PostFilterSearch(base_vecs, labels, **lsh_params_large)
    
    def search(self,
               query: np.ndarray,
               lo: int, hi: int,
               k: int = 10) -> np.ndarray:
        """Single-query search."""
        selectivity = (hi - lo + 1) / self.n_labels
        
        if selectivity <= self.tau_small:
            return self.exact_search.search(query, lo, hi, k)
        elif selectivity <= self.tau_medium:
            return self.lsh_medium.search(query, lo, hi, k)
        else:
            return self.lsh_large.search(query, lo, hi, k)
    
    def batch_search(self,
                     query_vecs: np.ndarray,
                     filter_ranges: np.ndarray,
                     k: int = 10) -> tuple[list[np.ndarray], float]:
        """
        Batch search routing queries to appropriate method by selectivity.
        
        Returns
        -------
        (results, total_search_time)
        """
        lo_vals = filter_ranges[:, 0]
        hi_vals = filter_ranges[:, 1]
        selectivities = (hi_vals - lo_vals + 1) / self.n_labels
        
        # Partition queries by selectivity
        small_mask = selectivities <= self.tau_small
        medium_mask = (selectivities > self.tau_small) & (selectivities <= self.tau_medium)
        large_mask = selectivities > self.tau_medium
        
        results = [None] * len(query_vecs)
        total_time = 0.0
        
        # Process small selectivity queries (exact search)
        small_indices = np.where(small_mask)[0]
        if len(small_indices) > 0:
            print(f"[Adaptive] {len(small_indices)} queries with small selectivity → exact search")
            small_queries = query_vecs[small_indices]
            small_ranges = filter_ranges[small_indices]
            small_results, small_time = self.exact_search.batch_search(small_queries, small_ranges, k)
            for i, idx in enumerate(small_indices):
                results[idx] = small_results[i]
            total_time += small_time
        
        # Process medium selectivity queries (filter-augmented LSH)
        medium_indices = np.where(medium_mask)[0]
        if len(medium_indices) > 0:
            print(f"[Adaptive] {len(medium_indices)} queries with medium selectivity → filter-augmented LSH")
            medium_queries = query_vecs[medium_indices]
            medium_ranges = filter_ranges[medium_indices]
            medium_results, medium_time = self.lsh_medium.batch_search(medium_queries, medium_ranges, k)
            for i, idx in enumerate(medium_indices):
                results[idx] = medium_results[i]
            total_time += medium_time
        
        # Process large selectivity queries (vector-dominant LSH)
        large_indices = np.where(large_mask)[0]
        if len(large_indices) > 0:
            print(f"[Adaptive] {len(large_indices)} queries with large selectivity → vector-dominant LSH")
            large_queries = query_vecs[large_indices]
            large_ranges = filter_ranges[large_indices]
            large_results, large_time = self.lsh_large.batch_search(large_queries, large_ranges, k)
            for i, idx in enumerate(large_indices):
                results[idx] = large_results[i]
            total_time += large_time
        
        return results, total_time
    
    def candidate_stats(self, query_vecs, filter_ranges):
        """Return candidate statistics across all branches."""
        # Simplified: just use medium branch stats
        return self.lsh_medium.candidate_stats(query_vecs, filter_ranges)
