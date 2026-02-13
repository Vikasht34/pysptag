"""
K-means clustering (baseline)
"""
import numpy as np
from typing import Tuple, List
from .base import ClusteringAlgorithm


class KMeansClustering(ClusteringAlgorithm):
    """Simple k-means clustering"""
    
    def __init__(self, max_iters: int = 20, metric: str = 'L2'):
        self.max_iters = max_iters
        self.metric = metric
    
    def cluster(
        self,
        data: np.ndarray,
        target_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-means clustering"""
        n, dim = data.shape
        k = min(target_clusters, n)
        
        # Initialize centroids
        indices = np.random.choice(n, k, replace=False)
        centroids = data[indices].copy()
        
        for iter in range(self.max_iters):
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
        
        return centroids, labels
    
    def assign_with_replicas(
        self,
        data: np.ndarray,
        centroids: np.ndarray,
        replica_count: int,
        posting_limit: int
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Assign vectors to multiple centroids with posting limits.
        CRITICAL: Only truncate if posting exceeds limit, don't pre-truncate.
        """
        n = len(data)
        k = len(centroids)
        
        # Find top-k nearest centroids
        if self.metric == 'L2':
            dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        else:
            dists = -np.dot(data, centroids.T)
        
        nearest = np.argsort(dists, axis=1)[:, :replica_count]
        
        # Build postings (collect all assignments first)
        postings = [[] for _ in range(k)]
        posting_dists = [[] for _ in range(k)]  # Track distances for sorting
        
        for vec_id, centroid_ids in enumerate(nearest):
            for idx, cid in enumerate(centroid_ids):
                postings[cid].append(vec_id)
                posting_dists[cid].append(dists[vec_id, cid])
        
        # Apply limits by keeping closest vectors
        replica_counts = np.zeros(n, dtype=int)
        truncated = 0
        
        for cid in range(k):
            if len(postings[cid]) > posting_limit:
                # Sort by distance, keep closest
                sorted_indices = np.argsort(posting_dists[cid])
                postings[cid] = [postings[cid][i] for i in sorted_indices[:posting_limit]]
                truncated += len(sorted_indices) - posting_limit
            
            for vec_id in postings[cid]:
                replica_counts[vec_id] += 1
        
        if truncated > 0:
            print(f"  Truncated {truncated} assignments, kept closest vectors")
        
        return postings, replica_counts
