"""
KD-Tree implementation for SPANN centroid search
Based on SPTAG's KDTree.h
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KDTNode:
    """KD-Tree node"""
    left: int = -1
    right: int = -1
    split_dim: int = -1
    split_value: float = 0.0


class KDTree:
    """KD-Tree for fast nearest neighbor search"""
    
    def __init__(self, num_trees: int = 1, samples: int = 100):
        self.num_trees = num_trees
        self.samples = samples
        self.tree_roots: List[KDTNode] = []
        self.tree_start: List[int] = []
        self.indices: np.ndarray = None
        
    def build(self, data: np.ndarray):
        """Build KD-Trees on data"""
        n, dim = data.shape
        
        # Build multiple trees
        self.tree_start = [0]
        for tree_idx in range(self.num_trees):
            # Random permutation for each tree
            perm = np.random.permutation(n)
            
            # Build tree
            root_idx = len(self.tree_roots)
            self.tree_start.append(root_idx)
            self._build_tree(data, perm, 0, n)
            
        self.indices = np.arange(n)
        
    def _build_tree(self, data: np.ndarray, indices: np.ndarray, 
                    start: int, end: int) -> int:
        """Recursively build KD-Tree"""
        if start >= end:
            return -1
            
        node_idx = len(self.tree_roots)
        node = KDTNode()
        
        if end - start <= 1:
            # Leaf node - store data index
            node.left = indices[start]  # Store actual data index
            node.right = -1
            self.tree_roots.append(node)
            return node_idx
            
        # Find split dimension (max variance)
        subset = data[indices[start:end]]
        variances = np.var(subset, axis=0)
        split_dim = np.argmax(variances)
        
        # Find split value (median)
        values = subset[:, split_dim]
        split_value = np.median(values)
        
        # Partition
        mid = start
        for i in range(start, end):
            if data[indices[i], split_dim] < split_value:
                indices[mid], indices[i] = indices[i], indices[mid]
                mid += 1
                
        # Ensure at least one element on each side
        if mid == start:
            mid = start + 1
        elif mid == end:
            mid = end - 1
            
        node.split_dim = split_dim
        node.split_value = split_value
        self.tree_roots.append(node)
        
        # Build children
        node.left = self._build_tree(data, indices, start, mid)
        node.right = self._build_tree(data, indices, mid, end)
        
        return node_idx
        
    def search(self, query: np.ndarray, data: np.ndarray, k: int,
               metric: str = 'L2') -> np.ndarray:
        """Search k nearest neighbors using KD-Trees"""
        candidates = set()
        
        # Search all trees
        for tree_idx in range(self.num_trees):
            root_idx = self.tree_start[tree_idx]
            if root_idx < len(self.tree_roots):
                self._search_tree(query, data, root_idx, candidates, k * 2, metric)
        
        if not candidates:
            return np.array([], dtype=np.int32)
            
        # Compute distances and return top-k
        candidates = list(candidates)
        if metric == 'L2':
            dists = np.sum((data[candidates] - query) ** 2, axis=1)
        elif metric in ('IP', 'Cosine'):
            dists = -np.dot(data[candidates], query)
        else:
            dists = np.zeros(len(candidates))
            
        top_k_idx = np.argsort(dists)[:k]
        return np.array([candidates[i] for i in top_k_idx], dtype=np.int32)
        
    def _search_tree(self, query: np.ndarray, data: np.ndarray,
                     node_idx: int, candidates: set, max_candidates: int,
                     metric: str):
        """Recursively search KD-Tree"""
        if node_idx < 0 or node_idx >= len(self.tree_roots):
            return
            
        node = self.tree_roots[node_idx]
        
        # Leaf node - add data index to candidates
        if node.right < 0:
            if len(candidates) < max_candidates and node.left >= 0:
                candidates.add(node.left)  # node.left stores data index
            return
            
        # Traverse tree
        if query[node.split_dim] < node.split_value:
            # Go left first
            if node.left >= 0:
                self._search_tree(query, data, node.left, candidates, max_candidates, metric)
            if node.right >= 0 and len(candidates) < max_candidates:
                self._search_tree(query, data, node.right, candidates, max_candidates, metric)
        else:
            # Go right first
            if node.right >= 0:
                self._search_tree(query, data, node.right, candidates, max_candidates, metric)
            if node.left >= 0 and len(candidates) < max_candidates:
                self._search_tree(query, data, node.left, candidates, max_candidates, metric)
