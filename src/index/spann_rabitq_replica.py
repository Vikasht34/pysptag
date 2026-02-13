"""
SPANN with RaBitQ - EXACT C++ implementation with replication
Key: Each vector assigned to multiple posting lists (replicaCount=8)
Supports L2, InnerProduct, and Cosine metrics
"""
import numpy as np
from typing import Tuple, Optional, Literal
from ..core.bktree import BKTree
from ..core.rng import RNG, MetricType
from ..quantization.rabitq import RaBitQ


class SPANNRaBitQReplica:
    """SPANN with RaBitQ and replication (exact C++ SPTAG)"""
    
    def __init__(
        self,
        dim: int,
        target_posting_size: int = 10000,
        replica_count: int = 8,  # C++ default
        num_trees: int = 1,
        kmeans_k: int = 32,
        bq: int = 4,
        use_rabitq: bool = True,  # Enable/disable RaBitQ quantization
        metric: MetricType = 'L2'  # Distance metric
    ):
        self.dim = dim
        self.target_posting_size = target_posting_size
        self.replica_count = replica_count
        self.use_rabitq = use_rabitq
        self.metric = metric
        self.bktree = BKTree(num_trees=num_trees, kmeans_k=kmeans_k)
        self.rng = RNG(neighborhood_size=32, metric=metric)
        self.bq = bq
        
        self.centroids: Optional[np.ndarray] = None
        self.posting_lists: list = []
        self.posting_codes: list = []
        self.posting_rabitqs: list = []
        self.num_clusters = 0
        
    def build(self, data: np.ndarray):
        """Build SPANN with replication + quantize posting lists"""
        n, dim = data.shape
        assert dim == self.dim
        
        print(f"Building SPANN+RaBitQ with {self.replica_count}× replication for {n} vectors")
        
        # Step 1: Cluster
        self.num_clusters = max(1, n // self.target_posting_size)
        print(f"[1/5] Clustering into {self.num_clusters} clusters...")
        labels, self.centroids = self._balanced_kmeans(data, self.num_clusters)
        
        # Step 2: Assign each vector to replica_count nearest centroids
        print(f"[2/5] Assigning vectors to {self.replica_count} nearest centroids...")
        self.posting_lists = [[] for _ in range(self.num_clusters)]
        
        for i in range(n):
            # Find replica_count nearest centroids (use correct metric)
            if self.metric == 'L2':
                dists = np.sum((self.centroids - data[i]) ** 2, axis=1)
            elif self.metric == 'IP':
                dists = -np.dot(self.centroids, data[i])
            elif self.metric == 'Cosine':
                dists = -np.dot(self.centroids, data[i])
            nearest = np.argsort(dists)[:self.replica_count]
            
            # Add to all replica posting lists
            for centroid_id in nearest:
                self.posting_lists[centroid_id].append(i)
        
        print(f"  Total assignments: {sum(len(p) for p in self.posting_lists):,}")
        print(f"  Avg replicas per vector: {sum(len(p) for p in self.posting_lists) / n:.1f}")
        
        # Step 3: Quantize each posting list (if enabled)
        print(f"[3/5] {'Quantizing' if self.use_rabitq else 'Storing'} posting lists...")
        self.posting_codes = []
        self.posting_rabitqs = []
        
        total_original = 0
        total_compressed = 0
        
        for i in range(self.num_clusters):
            posting_ids = np.array(self.posting_lists[i], dtype=np.int32)
            if len(posting_ids) == 0:
                self.posting_rabitqs.append(None)
                self.posting_codes.append(np.array([]))
                continue
                
            posting_vecs = data[posting_ids]
            
            if self.use_rabitq:
                # Quantize this posting
                rabitq = RaBitQ(dim=self.dim, bq=self.bq, metric=self.metric)
                codes = rabitq.build(posting_vecs)
                
                self.posting_rabitqs.append(rabitq)
                self.posting_codes.append(codes)
                
                total_original += posting_vecs.nbytes
                total_compressed += codes.nbytes
            else:
                # Store original vectors (no quantization)
                self.posting_rabitqs.append(None)
                self.posting_codes.append(posting_vecs)
                
                total_original += posting_vecs.nbytes
                total_compressed += posting_vecs.nbytes
        
        if self.use_rabitq:
            print(f"  Original: {total_original / 1024 / 1024:.2f} MB")
            print(f"  Compressed: {total_compressed / 1024 / 1024:.2f} MB")
            print(f"  Savings: {(1 - total_compressed / total_original) * 100:.1f}%")
        else:
            print(f"  Stored: {total_original / 1024 / 1024:.2f} MB (no compression)")
        
        # Step 4: Build BKTree + RNG on centroids
        print(f"[4/5] Building BKTree+RNG on centroids...")
        self.bktree.build(self.centroids)
        self.rng.build(self.centroids)
        
        msg = "quantization" if self.use_rabitq else "no quantization"
        print(f"✓ Index built with {self.replica_count}× replication + {msg}")
    
    def _balanced_kmeans(self, data: np.ndarray, k: int, max_iter: int = 50):
        """Fast vectorized balanced k-means"""
        n = len(data)
        center_idx = np.random.choice(n, k, replace=False)
        centers = data[center_idx].copy()
        lambda_penalty = 1.0 / n
        
        for iteration in range(max_iter):
            # Vectorized distance computation (batch processing)
            batch_size = 10000
            labels = np.zeros(n, dtype=np.int32)
            counts = np.zeros(k, dtype=np.int32)
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[start:end]
                
                # Compute distances for batch: (batch_size, k) - use correct metric
                if self.metric == 'L2':
                    dists = np.sum((batch[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                elif self.metric in ('IP', 'Cosine'):
                    dists = -np.dot(batch, centers.T)  # Negative for minimization
                dists += lambda_penalty * counts[None, :]
                
                batch_labels = np.argmin(dists, axis=1)
                labels[start:end] = batch_labels
                
                # Update counts
                for label in batch_labels:
                    counts[label] += 1
            
            # Recompute centers
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centers[j] = data[mask].mean(axis=0)
                else:
                    new_centers[j] = centers[j]
            
            # Check convergence
            diff = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if diff < 1e-3:
                print(f"  Converged at iteration {iteration+1}")
                break
            
            diff = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if diff < 1e-3:
                break
        
        return labels, centers
    
    def search(self, query: np.ndarray, data: np.ndarray, k: int = 10, 
               search_internal_result_num: int = 64, max_check: int = 4096):
        """Search using quantized postings with replication"""
        # Find nearest centroids (use correct metric)
        if self.metric == 'L2':
            centroid_dists = np.sum((self.centroids - query) ** 2, axis=1)
        elif self.metric == 'IP':
            centroid_dists = -np.dot(self.centroids, query)
        elif self.metric == 'Cosine':
            centroid_dists = -np.dot(self.centroids, query)
        nearest_centroids = np.argsort(centroid_dists)[:search_internal_result_num]
        
        # Search postings with deduplication
        seen = set()
        all_indices = []
        
        for centroid_id in nearest_centroids:
            posting_ids = self.posting_lists[centroid_id]
            if len(posting_ids) == 0:
                continue
            
            codes = self.posting_codes[centroid_id]
            rabitq = self.posting_rabitqs[centroid_id]
            
            # Limit k to posting size
            search_k = min(max_check, len(posting_ids))
            
            if self.use_rabitq:
                # Use RaBitQ for fast filtering (distances not comparable across postings)
                _, local_indices = rabitq.search(query, codes, None, k=search_k)
            else:
                # Direct distance computation (codes = original vectors)
                dists = np.sum((codes - query) ** 2, axis=1)
                local_indices = np.argsort(dists)[:search_k]
            
            # Map to global IDs and deduplicate
            for local_idx in local_indices:
                global_id = posting_ids[local_idx]
                if global_id not in seen:
                    seen.add(global_id)
                    all_indices.append(global_id)
                    
                    if len(all_indices) >= max_check:
                        break
            
            if len(all_indices) >= max_check:
                break
        
        # Rerank with true distances
        if len(all_indices) == 0:
            return np.array([]), np.array([])
        
        all_indices = np.array(all_indices)
        true_dists = np.sum((data[all_indices] - query) ** 2, axis=1)
        top_k_idx = np.argsort(true_dists)[:k]
        
        return true_dists[top_k_idx], all_indices[top_k_idx]
