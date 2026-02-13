"""
SPANN with disk-based posting lists for billion-scale datasets
Saves posting lists to disk, loads on-demand during search
"""
import numpy as np
import os
import pickle
from typing import Tuple, Optional
from ..core.bktree import BKTree
from ..core.rng import RNG, MetricType
from ..quantization.rabitq_numba import RaBitQNumba


class SPANNDiskBased:
    """SPANN with disk-based posting lists"""
    
    def __init__(
        self,
        dim: int,
        target_posting_size: int = 10000,
        replica_count: int = 8,
        bq: int = 4,
        metric: MetricType = 'L2',
        disk_path: str = './spann_index'
    ):
        self.dim = dim
        self.target_posting_size = target_posting_size
        self.replica_count = replica_count
        self.bq = bq
        self.metric = metric
        self.disk_path = disk_path
        
        # Create disk directory
        os.makedirs(disk_path, exist_ok=True)
        os.makedirs(os.path.join(disk_path, 'postings'), exist_ok=True)
        
    def build(self, data: np.ndarray):
        """Build index and save posting lists to disk"""
        n, dim = data.shape
        print(f"Building disk-based SPANN for {n} vectors")
        
        # Step 1: Clustering (simple k-means)
        print("[1/5] Clustering...")
        self.num_clusters = max(1, n // self.target_posting_size)
        
        # Initialize centroids randomly
        indices = np.random.choice(n, self.num_clusters, replace=False)
        self.centroids = data[indices].copy()
        
        # K-means iterations
        for iteration in range(20):
            # Assign to nearest centroid
            if self.metric == 'L2':
                dists = np.sum((data[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
            elif self.metric in ('IP', 'Cosine'):
                dists = -np.dot(data, self.centroids.T)
            
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                      else self.centroids[i] for i in range(self.num_clusters)])
            
            # Check convergence
            if np.allclose(new_centroids, self.centroids):
                print(f"  Converged at iteration {iteration + 1}")
                break
            
            self.centroids = new_centroids.astype(np.float32)
        
        # Step 2: Assign to multiple centroids (replication)
        print(f"[2/5] Assigning vectors to {self.replica_count} nearest centroids...")
        if self.metric == 'L2':
            dists = np.sum((data[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        elif self.metric in ('IP', 'Cosine'):
            dists = -np.dot(data, self.centroids.T)
        
        nearest_centroids = np.argsort(dists, axis=1)[:, :self.replica_count]
        
        # Build posting lists
        self.posting_lists = [[] for _ in range(self.num_clusters)]
        for vec_id, centroid_ids in enumerate(nearest_centroids):
            for centroid_id in centroid_ids:
                self.posting_lists[centroid_id].append(vec_id)
        
        # Step 3: Quantize and save posting lists to disk
        print("[3/5] Quantizing and saving posting lists to disk...")
        total_original = 0
        total_compressed = 0
        
        for i, posting_ids in enumerate(self.posting_lists):
            if len(posting_ids) == 0:
                continue
            
            posting_ids = np.array(posting_ids, dtype=np.int32)
            posting_vecs = data[posting_ids]
            
            # Quantize
            rabitq = RaBitQNumba(dim=self.dim, bq=self.bq, metric=self.metric)
            codes = rabitq.build(posting_vecs)
            
            # Save to disk
            posting_file = os.path.join(self.disk_path, 'postings', f'posting_{i}.pkl')
            with open(posting_file, 'wb') as f:
                pickle.dump({
                    'posting_ids': posting_ids,
                    'codes': codes,
                    'rabitq': rabitq
                }, f)
            
            total_original += posting_vecs.nbytes
            total_compressed += codes.nbytes + posting_ids.nbytes
            
            if (i + 1) % 50 == 0:
                print(f"  Saved {i+1}/{self.num_clusters} postings")
        
        print(f"  Original: {total_original/1024**2:.2f} MB")
        print(f"  Compressed: {total_compressed/1024**2:.2f} MB")
        print(f"  Savings: {(1 - total_compressed/total_original)*100:.1f}%")
        
        # Step 4: Build BKTree on centroids (for fast centroid search)
        print("[4/5] Building BKTree on centroids...")
        self.bktree = BKTree(self.centroids)
        
        # Step 5: Save metadata
        print("[5/5] Saving metadata...")
        metadata = {
            'dim': self.dim,
            'num_clusters': self.num_clusters,
            'replica_count': self.replica_count,
            'bq': self.bq,
            'metric': self.metric,
            'centroids': self.centroids,
            'bktree': self.bktree
        }
        with open(os.path.join(self.disk_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Index built and saved to {self.disk_path}")
    
    @classmethod
    def load(cls, disk_path: str):
        """Load index from disk"""
        print(f"Loading index from {disk_path}...")
        
        # Load metadata
        with open(os.path.join(disk_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        index = cls(
            dim=metadata['dim'],
            replica_count=metadata['replica_count'],
            bq=metadata['bq'],
            metric=metadata['metric'],
            disk_path=disk_path
        )
        
        index.num_clusters = metadata['num_clusters']
        index.centroids = metadata['centroids']
        index.bktree = metadata['bktree']
        
        print(f"✓ Index loaded: {index.num_clusters} clusters, {len(index.centroids)} centroids")
        return index
    
    def _load_posting(self, centroid_id: int):
        """Load posting list from disk"""
        posting_file = os.path.join(self.disk_path, 'postings', f'posting_{centroid_id}.pkl')
        if not os.path.exists(posting_file):
            return None, None, None
        
        with open(posting_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['posting_ids'], data['codes'], data['rabitq']
    
    def search(
        self,
        query: np.ndarray,
        data: np.ndarray,
        k: int = 10,
        search_internal_result_num: int = 64,
        max_check: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search with disk-based posting lists"""
        
        # Find nearest centroids (brute force is fast for <10K centroids)
        if self.metric == 'L2':
            centroid_dists = np.sum((self.centroids - query) ** 2, axis=1)
        elif self.metric == 'IP':
            centroid_dists = -np.dot(self.centroids, query)
        elif self.metric == 'Cosine':
            centroid_dists = -np.dot(self.centroids, query)
        
        nearest_centroids = np.argsort(centroid_dists)[:search_internal_result_num]
        
        # Search postings (load from disk on-demand)
        seen = set()
        all_indices = []
        
        for centroid_id in nearest_centroids:
            # Load posting from disk
            posting_ids, codes, rabitq = self._load_posting(centroid_id)
            if posting_ids is None:
                continue
            
            search_k = min(max_check, len(posting_ids))
            
            # Search posting
            _, local_indices = rabitq.search(query, codes, k=search_k)
            
            # Deduplicate
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
        
        candidates = data[all_indices]
        if self.metric == 'L2':
            final_dists = np.sum((candidates - query) ** 2, axis=1)
        elif self.metric == 'IP':
            final_dists = -np.dot(candidates, query)
        elif self.metric == 'Cosine':
            final_dists = -np.dot(candidates, query)
        
        top_k_local = np.argsort(final_dists)[:k]
        top_k_dists = final_dists[top_k_local]
        top_k_indices = np.array([all_indices[i] for i in top_k_local])
        
        return top_k_dists, top_k_indices
