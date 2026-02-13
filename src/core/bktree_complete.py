"""
Complete BKTree implementation following SPTAG's structure.
Direct port from SPTAG/AnnService/inc/Core/Common/BKTree.h
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BKTNode:
    """BKTree node structure (matches SPTAG)"""
    centerid: int = -1
    childStart: int = -1
    childEnd: int = -1


class BKTree:
    """
    Ball K-Tree implementation following SPTAG.
    Hierarchical clustering using k-means at each level.
    """
    
    def __init__(
        self,
        kmeans_k: int = 32,
        leaf_size: int = 8,
        num_trees: int = 1,
        samples: int = 1000,
        balance_factor: float = -1.0,
        metric: str = 'L2'
    ):
        """
        Args:
            kmeans_k: K for k-means at each level
            leaf_size: Min vectors to stop splitting
            num_trees: Number of trees (usually 1)
            samples: Samples for initialization
            balance_factor: Balance factor for splitting
            metric: Distance metric
        """
        self.kmeans_k = kmeans_k
        self.leaf_size = leaf_size
        self.num_trees = num_trees
        self.samples = samples
        self.balance_factor = balance_factor
        self.metric = metric
        
        # Tree structure
        self.nodes: List[BKTNode] = []
        self.tree_roots: List[int] = []  # Root node indices
        
    def build(self, data: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build BKTree from data.
        
        Args:
            data: (N, D) array of vectors
            indices: Optional index mapping
        """
        n, dim = data.shape
        
        if indices is None:
            indices = np.arange(n)
        
        self.nodes = []
        self.tree_roots = []
        
        # Build each tree on FULL dataset
        # Note: self.samples is used for k-means sampling, not tree size!
        for tree_id in range(self.num_trees):
            # Use all indices for tree building
            tree_indices = indices.copy()
            
            # Shuffle for randomness
            np.random.shuffle(tree_indices)
            
            # Build tree recursively on ALL data
            root_idx = self._build_node(data, tree_indices, 0, len(tree_indices))
            self.tree_roots.append(root_idx)
    
    def _build_node(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        first: int,
        last: int
    ) -> int:
        """
        Recursively build tree node with balanced k-means.
        
        Args:
            data: Full dataset
            indices: Index array for this subtree
            first: Start index in indices array
            last: End index in indices array
            
        Returns:
            Node index in self.nodes
        """
        count = last - first
        
        # Create leaf node if small enough
        if count <= self.leaf_size:
            center_id = indices[first]
            node = BKTNode(centerid=center_id)
            node_idx = len(self.nodes)
            self.nodes.append(node)
            return node_idx
        
        # Balanced k-means clustering
        k = min(self.kmeans_k, count)
        subset = data[indices[first:last]]
        
        # Use balanced k-means with auto-selected lambda
        from ..clustering.balanced_kmeans import balanced_kmeans, dynamic_factor_select
        
        if self.balance_factor < 0:
            # Auto-select lambda factor
            lambda_factor = dynamic_factor_select(subset, k, samples=min(1000, count), metric=self.metric)
        else:
            lambda_factor = self.balance_factor
        
        labels, centroids = balanced_kmeans(subset, k, lambda_factor, metric=self.metric)
        
        # Find actual center IDs (closest vectors to centroids)
        center_ids = []
        cluster_starts = []
        cluster_ends = []
        
        for i in range(k):
            mask = labels == i
            if not mask.any():
                continue
            
            cluster_data = subset[mask]
            cluster_indices = indices[first:last][mask]
            
            # Find closest to centroid
            if self.metric == 'L2':
                dists = np.sum((cluster_data - centroids[i]) ** 2, axis=1)
            elif self.metric in ('IP', 'Cosine'):
                dists = -np.dot(cluster_data, centroids[i])
            else:
                dists = np.sum((cluster_data - centroids[i]) ** 2, axis=1)
            
            closest_idx = np.argmin(dists)
            center_ids.append(cluster_indices[closest_idx])
            
            # Track cluster boundaries
            cluster_starts.append(first + np.where(labels == i)[0][0])
            cluster_ends.append(first + np.where(labels == i)[0][-1] + 1)
        
        # Create node
        if not center_ids:
            center_id = indices[first]
            node = BKTNode(centerid=center_id)
            node_idx = len(self.nodes)
            self.nodes.append(node)
            return node_idx
        
        center_id = center_ids[0]
        node = BKTNode(centerid=center_id)
        node_idx = len(self.nodes)
        self.nodes.append(node)
        
        # Recursively build children
        if len(center_ids) > 1:
            # Rearrange indices by cluster
            sorted_indices = []
            for i in range(k):
                mask = labels == i
                if mask.any():
                    sorted_indices.extend(indices[first:last][mask])
            indices[first:last] = sorted_indices
            
            # Build child nodes
            node.childStart = len(self.nodes)
            child_start = first
            
            for i in range(k):
                mask = labels == i
                if not mask.any():
                    continue
                
                child_end = child_start + mask.sum()
                child_idx = self._build_node(data, indices, child_start, child_end)
                child_start = child_end
            
            node.childEnd = len(self.nodes)
        
        return node_idx
    
    def _kmeans(
        self,
        data: np.ndarray,
        k: int,
        max_iters: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-means clustering (simplified version of SPTAG's KmeansAssign).
        
        Args:
            data: (N, D) array
            k: Number of clusters
            max_iters: Max iterations
            
        Returns:
            labels: (N,) cluster assignments
            centroids: (K, D) cluster centers
        """
        n, dim = data.shape
        k = min(k, n)
        
        # Initialize centroids (k-means++)
        centroids = np.zeros((k, dim), dtype=data.dtype)
        centroids[0] = data[np.random.randint(n)]
        
        for i in range(1, k):
            dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids[:i]], axis=0)
            probs = dists / dists.sum()
            centroids[i] = data[np.random.choice(n, p=probs)]
        
        # Iterate
        for _ in range(max_iters):
            # Assign to nearest centroid
            if self.metric == 'L2':
                dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            else:  # IP/Cosine
                dists = -np.dot(data, centroids.T)
            
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return labels, centroids
    
    def search(
        self,
        query: np.ndarray,
        data: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search BKTree for k nearest neighbors.
        
        Args:
            query: (D,) query vector
            data: (N, D) full dataset
            k: Number of neighbors
            
        Returns:
            distances: (k,) distances
            indices: (k,) indices
        """
        # Simple greedy search from root
        candidates = []
        
        for root_idx in self.tree_roots:
            self._search_node(query, data, root_idx, candidates, k * 10)
        
        # Sort and return top-k
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:k]
        
        if not candidates:
            return np.array([]), np.array([])
        
        dists = np.array([c[0] for c in candidates])
        indices = np.array([c[1] for c in candidates])
        
        return dists, indices
    
    def _search_node(
        self,
        query: np.ndarray,
        data: np.ndarray,
        node_idx: int,
        candidates: List[Tuple[float, int]],
        max_candidates: int
    ):
        """Recursively search tree node"""
        if node_idx >= len(self.nodes):
            return
        
        node = self.nodes[node_idx]
        
        # Check center
        center_vec = data[node.centerid]
        dist = np.sum((query - center_vec) ** 2)
        candidates.append((dist, node.centerid))
        
        # Search children
        if node.childStart >= 0:
            for child_idx in range(node.childStart, node.childEnd):
                if len(candidates) < max_candidates:
                    self._search_node(query, data, child_idx, candidates, max_candidates)
