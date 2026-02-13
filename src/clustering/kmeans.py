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
        target_clusters: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-means clustering.
        If target_clusters is None, uses ratio-based approach (1% of data).
        """
        n, dim = data.shape
        
        if target_clusters is None:
            # SPTAG approach: 1% of data as centroids
            k = max(10, int(n * 0.01))
        else:
            k = min(target_clusters, n)
        
        # Initialize centroids (k-means++)
        centroids = np.zeros((k, dim), dtype=data.dtype)
        centroids[0] = data[np.random.randint(n)]
        
        for i in range(1, k):
            dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids[:i]], axis=0)
            probs = dists / dists.sum()
            centroids[i] = data[np.random.choice(n, p=probs)]
        
        # Iterate
        for _ in range(self.max_iters):
            # Assign
            if self.metric == 'L2':
                dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            else:
                dists = -np.dot(data, centroids.T)
            
            labels = np.argmin(dists, axis=1)
            
            # Update
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if mask.any():
                    new_centroids[i] = data[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
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
        Assign vectors to multiple centroids with SPTAG-style posting limits.
        Only truncate if expected posting size exceeds limit.
        """
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
        posting_dists = [[] for _ in range(k)]
        
        for vec_id, centroid_ids in enumerate(nearest):
            for cid in centroid_ids:
                postings[cid].append(vec_id)
                posting_dists[cid].append(dists[vec_id, cid])
        
        # SPTAG approach: Only truncate if expected size > limit
        expected_posting_size = (n * replica_count) // k if k > 0 else n
        should_truncate = expected_posting_size > posting_limit
        
        replica_counts = np.zeros(n, dtype=int)
        truncated = 0
        
        if should_truncate:
            print(f"  Expected posting size: {expected_posting_size}, limit: {posting_limit}")
            print(f"  Truncating to keep closest {posting_limit} vectors per posting")
            
            for cid in range(k):
                if len(postings[cid]) > posting_limit:
                    # Sort by distance, keep closest
                    sorted_indices = np.argsort(posting_dists[cid])
                    postings[cid] = [postings[cid][i] for i in sorted_indices[:posting_limit]]
                    truncated += len(sorted_indices) - posting_limit
        
        for cid in range(k):
            for vec_id in postings[cid]:
                replica_counts[vec_id] += 1
        
        if truncated > 0:
            print(f"  Truncated {truncated} assignments ({truncated/(n*replica_count)*100:.1f}%)")
        
        # Stats
        sizes = [len(p) for p in postings if len(p) > 0]
        if sizes:
            print(f"  Posting sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        return postings, replica_counts
