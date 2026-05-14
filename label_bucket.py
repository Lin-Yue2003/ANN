"""
label_bucket.py
===============
Label bucket index for accelerated exact small-range search.

Pre-computes label-to-indices mapping with efficient slicing.
"""

import numpy as np


class LabelBucketIndex:
    """
    Fast label-based filtering via pre-sorted index.
    
    Instead of:
        mask = (labels >= lo) & (labels <= hi)
        valid_ids = np.where(mask)[0]
    
    Use:
        valid_ids = index[lo:hi+1]  (fast O(1) slice)
    
    Parameters
    ----------
    labels : (N,) int32
        Label for each vector
    n_labels : int
        Total number of distinct labels (0 .. n_labels-1)
    """
    
    def __init__(self, labels: np.ndarray, n_labels: int):
        self.n_labels = n_labels
        self.labels = labels.astype(np.int32)
        
        # Sort indices by label
        self.sort_order = np.argsort(labels, kind='stable')
        sorted_labels = labels[self.sort_order]
        
        # For each label, store start/end indices in sort_order
        self.label_start = np.zeros(n_labels, dtype=np.int32)
        self.label_end = np.zeros(n_labels, dtype=np.int32)
        
        current_label = -1
        for pos, label in enumerate(sorted_labels):
            if label != current_label:
                if current_label >= 0:
                    self.label_end[current_label] = pos - 1
                self.label_start[label] = pos
                current_label = label
        if current_label >= 0:
            self.label_end[current_label] = len(sorted_labels) - 1
    
    def query_range(self, lo: int, hi: int) -> np.ndarray:
        """
        Fast range query: return all indices with labels in [lo, hi].
        
        Parameters
        ----------
        lo, hi : int
            Label range (inclusive)
        
        Returns
        -------
        (M,) int32 array of vector indices with labels in [lo, hi]
        """
        # Clamp to valid label range
        lo = max(0, min(lo, self.n_labels - 1))
        hi = max(0, min(hi, self.n_labels - 1))
        
        if lo > hi:
            return np.array([], dtype=np.int32)
        
        # Find contiguous range in sorted_order
        start_pos = self.label_start[lo]
        end_pos = self.label_end[hi]
        
        if start_pos > end_pos:
            return np.array([], dtype=np.int32)
        
        # Slice and return
        return self.sort_order[start_pos : end_pos + 1].astype(np.int32)
