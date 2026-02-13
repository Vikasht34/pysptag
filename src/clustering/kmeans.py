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
        """Assign vectors to multiple centroids with optional posting limits"""
        n = len(data)
        k = len(centroids)
        
        # Find top-k nearest centroids
        if self.metric == 'L2':
            dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        else:
            dists = -np.dot(data, centroids.T)
        
        nearest = np.argsort(dists, axis=1)[:, :replica_count]
        
        # Build postings
        postings = [[] for _ in range(k)]
        for vec_id, centroid_ids in enumerate(nearest):
            for cid in centroid_ids:
                postings[cid].append(vec_id)
        
        # Count replicas (no truncation for now - posting limits need more work)
        replica_counts = np.zeros(n, dtype=int)
        for cid in range(k):
            for vec_id in postings[cid]:
                replica_counts[vec_id] += 1
        
        return postings, replica_counts
