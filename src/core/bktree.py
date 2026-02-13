"""
BKTree - Ball Tree with K-means clustering
Direct Python port from SPTAG C++ implementation
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BKTNode:
    """Node in BKTree"""
    centerid: int = -1
    childStart: int = -1
    childEnd: int = -1


class BKTree:
    """
    Ball Tree with K-means clustering
    Exact port from SPTAG/AnnService/inc/Core/Common/BKTree.h
    """
    
    def __init__(
        self,
        num_trees: int = 1,
        kmeans_k: int = 32,
        leaf_size: int = 8,
        samples: int = 1000,
        balance_factor: float = -1.0
    ):
        self.num_trees = num_trees
        self.kmeans_k = kmeans_k
        self.leaf_size = leaf_size
        self.samples = samples
        self.balance_factor = balance_factor
        
        self.tree_roots: List[BKTNode] = []
        self.tree_start: List[int] = []
        self.sample_center_map: dict = {}
        
    def build(self, data: np.ndarray, indices: Optional[np.ndarray] = None):
        """
        Build BKTree from data
        
        Args:
            data: (N, D) array
            indices: Optional index mapping
        """
        n, dim = data.shape
        
        if indices is None:
            indices = np.arange(n, dtype=np.int32)
        else:
            indices = indices.copy()
            
        # Auto-select balance factor if needed
        if self.balance_factor < 0:
            self.balance_factor = self._dynamic_factor_select(data, indices)
            
        self.sample_center_map.clear()
        
        # Build multiple trees
        for tree_idx in range(self.num_trees):
            np.random.shuffle(indices)
            
            self.tree_start.append(len(self.tree_roots))
            self.tree_roots.append(BKTNode(centerid=n))
            
            # Stack for recursive building
            stack = [(self.tree_start[tree_idx], 0, n)]
            
            while stack:
                node_idx, first, last = stack.pop()
                self.tree_roots[node_idx].childStart = len(self.tree_roots)
                
                if last - first <= self.leaf_size:
                    # Leaf node - store all points
                    for j in range(first, last):
                        self.tree_roots.append(BKTNode(centerid=indices[j]))
                else:
                    # Internal node - cluster and recurse
                    labels, counts, centers = self._kmeans_clustering(
                        data, indices, first, last
                    )
                    
                    if len(counts) <= 1:
                        # Failed clustering - make leaf
                        sorted_idx = np.argsort(indices[first:last])
                        indices[first:last] = indices[first:last][sorted_idx]
                        
                        self.tree_roots[node_idx].centerid = indices[first]
                        self.tree_roots[node_idx].childStart = -self.tree_roots[node_idx].childStart
                        
                        for j in range(first + 1, last):
                            cid = indices[j]
                            self.tree_roots.append(BKTNode(centerid=cid))
                            self.sample_center_map[cid] = self.tree_roots[node_idx].centerid
                        
                        self.sample_center_map[-1 - self.tree_roots[node_idx].centerid] = node_idx
                    else:
                        # Create child nodes
                        pos = first
                        for k in range(len(counts)):
                            if counts[k] == 0:
                                continue
                            
                            cid = indices[pos + counts[k] - 1]
                            self.tree_roots.append(BKTNode(centerid=cid))
                            
                            if counts[k] > 1:
                                stack.append((len(self.tree_roots) - 1, pos, pos + counts[k] - 1))
                            
                            pos += counts[k]
                
                self.tree_roots[node_idx].childEnd = len(self.tree_roots)
    
    def _kmeans_clustering(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        first: int,
        last: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Balanced k-means clustering
        
        Returns:
            labels: Cluster assignment for each point
            counts: Number of points per cluster
            centers: Cluster centers
        """
        subset = data[indices[first:last]]
        n_samples = last - first
        k = min(self.kmeans_k, n_samples // self.leaf_size + 1)
        k = max(k, 2)
        k = min(k, n_samples)  # Can't have more clusters than samples
        
        # Initialize centers randomly
        center_idx = np.random.choice(n_samples, k, replace=False)
        centers = subset[center_idx].copy()
        
        # Balanced k-means with lambda penalty
        lambda_penalty = self.balance_factor / n_samples
        
        best_labels = None
        best_inertia = float('inf')
        
        for iter_num in range(100):
            # Assignment with balance penalty
            counts = np.zeros(k, dtype=np.int32)
            labels = np.zeros(n_samples, dtype=np.int32)
            
            for i in range(n_samples):
                dists = np.sum((subset[i] - centers) ** 2, axis=1)
                dists += lambda_penalty * counts  # Balance penalty
                labels[i] = np.argmin(dists)
                counts[labels[i]] += 1
            
            # Check convergence
            inertia = 0
            for i in range(n_samples):
                inertia += np.sum((subset[i] - centers[labels[i]]) ** 2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
            
            # Update centers
            new_centers = np.zeros_like(centers)
            new_counts = np.zeros(k, dtype=np.int32)
            
            for i in range(n_samples):
                new_centers[labels[i]] += subset[i]
                new_counts[labels[i]] += 1
            
            # Handle empty clusters
            for j in range(k):
                if new_counts[j] > 0:
                    new_centers[j] /= new_counts[j]
                else:
                    # Reinitialize from largest cluster
                    max_cluster = np.argmax(new_counts)
                    new_centers[j] = centers[max_cluster]
            
            # Check convergence
            diff = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            
            if diff < 1e-3:
                break
        
        # Final assignment without penalty
        counts = np.zeros(k, dtype=np.int32)
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            dists = np.sum((subset[i] - centers) ** 2, axis=1)
            labels[i] = np.argmin(dists)
            counts[labels[i]] += 1
        
        # Reorder indices by cluster
        sorted_idx = np.argsort(labels)
        indices[first:last] = indices[first:last][sorted_idx]
        
        # Remove empty clusters
        valid = counts > 0
        counts = counts[valid]
        centers = centers[valid]
        
        return labels, counts, centers
    
    def _dynamic_factor_select(self, data: np.ndarray, indices: np.ndarray) -> float:
        """Select balance factor dynamically"""
        # Try different factors and pick best
        best_factor = 100.0
        best_std = float('inf')
        
        for factor in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            self.balance_factor = factor
            _, counts, _ = self._kmeans_clustering(data, indices, 0, len(indices))
            
            if len(counts) > 0:
                std = np.std(counts) / np.mean(counts)
                if std < best_std:
                    best_std = std
                    best_factor = factor
        
        return best_factor
    
    def search(
        self,
        query: np.ndarray,
        data: np.ndarray,
        k: int,
        max_check: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search BKTree for k nearest neighbors
        
        Args:
            query: (D,) query vector
            data: (N, D) dataset
            k: Number of neighbors
            max_check: Max nodes to check (-1 = no limit)
            
        Returns:
            distances: (k,) distances
            indices: (k,) indices
        """
        if max_check < 0:
            max_check = len(self.tree_roots)
        
        # Priority queue: (distance, node_idx)
        candidates = []
        checked = 0
        
        # Search all trees
        for tree_idx in range(self.num_trees):
            root_idx = self.tree_start[tree_idx]
            stack = [(0.0, root_idx)]
            
            while stack and checked < max_check:
                dist, node_idx = stack.pop()
                node = self.tree_roots[node_idx]
                checked += 1
                
                if node.childStart < 0:
                    # Leaf node
                    child_start = -node.childStart
                    child_end = node.childEnd
                    
                    for i in range(child_start, child_end):
                        cid = self.tree_roots[i].centerid
                        d = np.sum((query - data[cid]) ** 2)
                        candidates.append((d, cid))
                else:
                    # Internal node - add children
                    for i in range(node.childStart, node.childEnd):
                        child = self.tree_roots[i]
                        if child.centerid >= 0:
                            d = np.sum((query - data[child.centerid]) ** 2)
                            stack.append((d, i))
        
        # Get top-k
        candidates.sort()
        candidates = candidates[:k]
        
        distances = np.array([d for d, _ in candidates])
        indices = np.array([idx for _, idx in candidates])
        
        return distances, indices
