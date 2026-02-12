"""
SPANN with RaBitQ - EXACT C++ implementation with replication
Key: Each vector assigned to multiple posting lists (replicaCount=8)
"""
import numpy as np
from typing import Tuple, Optional
from ..core.bktree import BKTree
from ..core.rng import RNG
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
        bq: int = 4
    ):
        self.dim = dim
        self.target_posting_size = target_posting_size
        self.replica_count = replica_count
        self.bktree = BKTree(num_trees=num_trees, kmeans_k=kmeans_k)
        self.rng = RNG(neighborhood_size=32)
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
            # Find replica_count nearest centroids
            dists = np.sum((self.centroids - data[i]) ** 2, axis=1)
            nearest = np.argsort(dists)[:self.replica_count]
            
            # Add to all replica posting lists
            for centroid_id in nearest:
                self.posting_lists[centroid_id].append(i)
        
        print(f"  Total assignments: {sum(len(p) for p in self.posting_lists):,}")
        print(f"  Avg replicas per vector: {sum(len(p) for p in self.posting_lists) / n:.1f}")
        
        # Step 3: Quantize each posting list
        print(f"[3/5] Quantizing posting lists with RaBitQ...")
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
            
            # Quantize this posting
            rabitq = RaBitQ(dim=self.dim, bq=self.bq)
            codes = rabitq.build(posting_vecs)
            
            self.posting_rabitqs.append(rabitq)
            self.posting_codes.append(codes)
            
            total_original += posting_vecs.nbytes
            total_compressed += codes.nbytes
        
        print(f"  Original: {total_original / 1024 / 1024:.2f} MB")
        print(f"  Compressed: {total_compressed / 1024 / 1024:.2f} MB")
        print(f"  Savings: {(1 - total_compressed / total_original) * 100:.1f}%")
        
        # Step 4: Build BKTree + RNG on centroids
        print(f"[4/5] Building BKTree+RNG on centroids...")
        self.bktree.build(self.centroids)
        self.rng.build(self.centroids)
        
        print(f"✓ Index built with {self.replica_count}× replication + quantization")
    
    def _balanced_kmeans(self, data: np.ndarray, k: int, max_iter: int = 100):
        n = len(data)
        center_idx = np.random.choice(n, k, replace=False)
        centers = data[center_idx].copy()
        lambda_penalty = 1.0 / n
        
        for _ in range(max_iter):
            counts = np.zeros(k, dtype=np.int32)
            labels = np.zeros(n, dtype=np.int32)
            
            for i in range(n):
                dists = np.sum((data[i] - centers) ** 2, axis=1)
                dists += lambda_penalty * counts
                labels[i] = np.argmin(dists)
                counts[labels[i]] += 1
            
            new_centers = np.zeros_like(centers)
            new_counts = np.zeros(k, dtype=np.int32)
            
            for i in range(n):
                new_centers[labels[i]] += data[i]
                new_counts[labels[i]] += 1
            
            for j in range(k):
                if new_counts[j] > 0:
                    new_centers[j] /= new_counts[j]
                else:
                    new_centers[j] = centers[j]
            
            diff = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if diff < 1e-3:
                break
        
        return labels, centers
    
    def search(self, query: np.ndarray, data: np.ndarray, k: int = 10, 
               search_internal_result_num: int = 64, max_check: int = 4096):
        """Search using quantized postings with replication"""
        # Find nearest centroids
        centroid_dists = np.sum((self.centroids - query) ** 2, axis=1)
        nearest_centroids = np.argsort(centroid_dists)[:search_internal_result_num]
        
        # Search quantized postings with deduplication
        seen = set()
        all_dists = []
        all_indices = []
        
        for centroid_id in nearest_centroids:
            posting_ids = self.posting_lists[centroid_id]
            if len(posting_ids) == 0:
                continue
            
            # Search quantized codes
            codes = self.posting_codes[centroid_id]
            rabitq = self.posting_rabitqs[centroid_id]
            posting_vecs = data[posting_ids]
            
            dists, local_indices = rabitq.search(query, codes, posting_vecs, k=max_check)
            
            # Map to global IDs and deduplicate
            for dist, local_idx in zip(dists, local_indices):
                global_id = posting_ids[local_idx]
                if global_id not in seen:
                    seen.add(global_id)
                    all_dists.append(dist)
                    all_indices.append(global_id)
                    
                    if len(all_dists) >= max_check:
                        break
            
            if len(all_dists) >= max_check:
                break
        
        # Get top-k
        if len(all_dists) == 0:
            return np.array([]), np.array([])
        
        top_k_idx = np.argsort(all_dists)[:k]
        return np.array(all_dists)[top_k_idx], np.array(all_indices)[top_k_idx]
