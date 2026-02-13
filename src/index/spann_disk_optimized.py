"""
Optimized disk-based SPANN with pluggable clustering algorithms.
Supports k-means and hierarchical (SPTAG-style) clustering.
"""
import numpy as np
import os
import struct
import mmap
from typing import Tuple, Optional, Dict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from ..core.rng import RNG, MetricType
from ..quantization.rabitq_numba import RaBitQNumba
from ..clustering import ClusteringAlgorithm, KMeansClustering, HierarchicalClustering


class SPANNDiskOptimized:
    """Optimized disk-based SPANN with pluggable clustering"""
    
    def __init__(
        self,
        dim: int,
        target_posting_size: int = 118,
        replica_count: int = 8,
        bq: int = 4,
        use_rabitq: bool = True,
        metric: MetricType = 'L2',
        tree_type: str = 'KDT',
        disk_path: str = './spann_index',
        cache_size: int = 128,
        num_threads: int = 1,
        clustering: str = 'hierarchical'  # 'kmeans' or 'hierarchical'
    ):
        self.dim = dim
        self.target_posting_size = target_posting_size
        self.replica_count = replica_count
        self.bq = bq
        self.use_rabitq = use_rabitq
        self.metric = metric
        self.tree_type = tree_type
        self.disk_path = disk_path
        self.cache_size = cache_size
        self.num_threads = num_threads
        
        # Select clustering algorithm
        if clustering == 'hierarchical':
            self.clusterer = HierarchicalClustering(
                select_threshold=6,
                split_threshold=25,
                ratio=0.01,
                kmeans_k=32,
                leaf_size=8,
                metric=metric
            )
        else:  # kmeans
            self.clusterer = KMeansClustering(metric=metric)
        
        # Create tree
        if tree_type == 'BKT':
            from ..core.bktree import BKTree
            self.tree = BKTree(num_trees=1, kmeans_k=32)
        else:  # KDT
            from ..core.kdtree import KDTree
            self.tree = KDTree(num_trees=1)
            
        self.rng = RNG(neighborhood_size=32, metric=metric)
        
        # Cache for posting lists
        self._posting_cache: Dict[int, Tuple] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Create disk directory
        os.makedirs(disk_path, exist_ok=True)
        os.makedirs(os.path.join(disk_path, 'postings'), exist_ok=True)
    
    def build(self, data: np.ndarray):
        """Build index with pluggable clustering"""
        n, dim = data.shape
        print(f"Building optimized SPANN for {n} vectors")
        print(f"  Clustering: {self.clusterer.__class__.__name__}")
        print(f"  Posting limit: {self.target_posting_size} vectors")
        print(f"  Replica count: {self.replica_count}")
        
        # Step 1: Cluster data
        print("[1/5] Clustering...")
        target_clusters = max(1, n // self.target_posting_size)
        self.centroids, labels = self.clusterer.cluster(data, target_clusters)
        self.num_clusters = len(self.centroids)
        
        # Step 2: Assign with replicas and posting limits
        print(f"[2/5] Assigning vectors to {self.num_clusters} centroids...")
        self.posting_lists, replica_counts = self.clusterer.assign_with_replicas(
            data, self.centroids, self.replica_count, self.target_posting_size
        )
        print(f"  Clustering: {self.clusterer.__class__.__name__}")
        print(f"  Posting limit: {self.target_posting_size} vectors")
        print(f"  Replica count: {self.replica_count}")
        
        # Step 1: Cluster data
        print("[1/5] Clustering...")
        target_clusters = max(1, n // self.target_posting_size)
        self.centroids, labels = self.clusterer.cluster(data, target_clusters)
        self.num_clusters = len(self.centroids)
        
        # Step 2: Assign with replicas and posting limits
        print(f"[2/5] Assigning vectors to {self.num_clusters} centroids...")
        self.posting_lists, replica_counts = self.clusterer.assign_with_replicas(
            data, self.centroids, self.replica_count, self.target_posting_size
        )
        
        # Step 3: Save posting lists in binary format (mmap-friendly)
        print("[3/5] Saving posting lists in binary format...")
        total_original = 0
        total_compressed = 0
        
        for i, posting_ids in enumerate(self.posting_lists):
            if len(posting_ids) == 0:
                continue
            
            posting_ids = np.array(posting_ids, dtype=np.int32)
            posting_vecs = data[posting_ids]
            
            if self.use_rabitq:
                # Quantize
                rabitq = RaBitQNumba(dim=self.dim, bq=self.bq, metric=self.metric)
                codes = rabitq.build(posting_vecs)
                
                # Save in binary format
                self._save_posting_binary(i, posting_ids, codes, rabitq)
                
                total_original += posting_vecs.nbytes
                total_compressed += codes.nbytes + posting_ids.nbytes
            else:
                # No quantization
                self._save_posting_binary(i, posting_ids, posting_vecs, None)
                
                total_original += posting_vecs.nbytes
                total_compressed += posting_vecs.nbytes + posting_ids.nbytes
            
            if (i + 1) % 50 == 0:
                print(f"  Saved {i+1}/{self.num_clusters} postings")
        
        if self.use_rabitq:
            print(f"  Original: {total_original/1024**2:.2f} MB")
            print(f"  Compressed: {total_compressed/1024**2:.2f} MB")
            print(f"  Savings: {(1 - total_compressed/total_original)*100:.1f}%")
        else:
            print(f"  Stored: {total_original/1024**2:.2f} MB (no compression)")
        
        # Step 4: Build tree + RNG
        print(f"[4/5] Building {self.tree_type}+RNG on centroids...")
        self.tree.build(self.centroids)
        self.rng.build(self.centroids)
        
        # Step 5: Save metadata
        print("[5/5] Saving metadata...")
        import pickle
        metadata = {
            'dim': self.dim,
            'num_clusters': self.num_clusters,
            'replica_count': self.replica_count,
            'bq': self.bq,
            'use_rabitq': self.use_rabitq,
            'metric': self.metric,
            'tree_type': self.tree_type,
            'centroids': self.centroids,
            'tree': self.tree,
            'rng': self.rng
        }
        with open(os.path.join(self.disk_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Index built and saved to {self.disk_path}")
    
    def _save_posting_binary(self, centroid_id: int, posting_ids: np.ndarray, 
                            codes: np.ndarray, rabitq: Optional[RaBitQNumba]):
        """Save posting in binary format for mmap"""
        posting_file = os.path.join(self.disk_path, 'postings', f'posting_{centroid_id}.bin')
        
        with open(posting_file, 'wb') as f:
            # Header
            f.write(struct.pack('III', len(posting_ids), codes.shape[1] if len(codes.shape) > 1 else codes.shape[0], 
                              1 if rabitq is None else 0))
            
            # Data
            posting_ids.tofile(f)
            codes.tofile(f)
            
            # RaBitQ params (if quantized)
            if rabitq is not None:
                import pickle
                rabitq_bytes = pickle.dumps(rabitq)
                f.write(struct.pack('I', len(rabitq_bytes)))
                f.write(rabitq_bytes)
    
    def _load_posting_mmap(self, centroid_id: int):
        """Load posting with memory-mapped file (zero-copy)"""
        # Check cache first
        if centroid_id in self._posting_cache:
            self._cache_hits += 1
            return self._posting_cache[centroid_id]
        
        self._cache_misses += 1
        
        posting_file = os.path.join(self.disk_path, 'postings', f'posting_{centroid_id}.bin')
        if not os.path.exists(posting_file):
            return None, None, None
        
        with open(posting_file, 'rb') as f:
            # Read header
            num_vecs, code_dim, is_unquantized = struct.unpack('III', f.read(12))
            
            # Memory-map the data section
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read posting IDs
            offset = 12
            posting_ids = np.frombuffer(mm, dtype=np.int32, count=num_vecs, offset=offset)
            offset += num_vecs * 4
            
            # Read codes
            if is_unquantized:
                codes = np.frombuffer(mm, dtype=np.float32, count=num_vecs * code_dim, offset=offset)
                codes = codes.reshape(num_vecs, code_dim)
                rabitq = None
            else:
                code_size = num_vecs * code_dim
                codes = np.frombuffer(mm, dtype=np.uint8, count=code_size, offset=offset)
                codes = codes.reshape(num_vecs, code_dim)
                offset += code_size
                
                # Read RaBitQ
                rabitq_size = struct.unpack('I', mm[offset:offset+4])[0]
                offset += 4
                import pickle
                rabitq = pickle.loads(mm[offset:offset+rabitq_size])
        
        # Cache result (keep only recent cache_size items)
        if len(self._posting_cache) >= self.cache_size:
            # Remove oldest (simple FIFO, could use LRU)
            self._posting_cache.pop(next(iter(self._posting_cache)))
        
        self._posting_cache[centroid_id] = (posting_ids.copy(), codes.copy(), rabitq)
        return self._posting_cache[centroid_id]
    
    def _load_postings_batch(self, centroid_ids: list):
        """Batch load multiple postings"""
        postings = {}
        uncached = []
        
        # Check cache first
        for cid in centroid_ids:
            if cid in self._posting_cache:
                postings[cid] = self._posting_cache[cid]
                self._cache_hits += 1
            else:
                uncached.append(cid)
        
        # Load uncached in batch
        for cid in uncached:
            result = self._load_posting_mmap(cid)
            if result[0] is not None:
                postings[cid] = result
        
        return postings
    
    def _fast_kmeans(self, data: np.ndarray, k: int, max_iter: int = 50):
        """Fast vectorized k-means"""
        n, dim = data.shape
        indices = np.random.choice(n, k, replace=False)
        centers = data[indices].copy()
        
        batch_size = 10000
        
        for iteration in range(max_iter):
            labels = np.zeros(n, dtype=np.int32)
            
            if self.metric == 'L2':
                centers_sq = np.sum(centers ** 2, axis=1)
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[start:end]
                
                if self.metric == 'L2':
                    batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)
                    dists = batch_sq + centers_sq - 2 * np.dot(batch, centers.T)
                elif self.metric in ('IP', 'Cosine'):
                    dists = -np.dot(batch, centers.T)
                
                labels[start:end] = np.argmin(dists, axis=1)
            
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centers[j] = data[mask].mean(axis=0)
                else:
                    new_centers[j] = centers[j]
            
            diff = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            if diff < 1e-2 * k:
                print(f"  Converged at iteration {iteration+1}")
                break
        
        return labels, centers
    
    def search(
        self,
        query: np.ndarray,
        data: np.ndarray,
        k: int = 10,
        search_internal_result_num: int = 64,
        max_check: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized search with batch loading"""
        
        # Find nearest centroids
        if hasattr(self, 'rng') and len(self.rng.graph) > 0:
            if self.tree_type == 'BKT':
                from ..core.bktree_rng_search import bktree_rng_search
                nearest_centroids = bktree_rng_search(
                    query, self.centroids, self.tree.tree_roots,
                    self.tree.tree_start, self.rng.graph,
                    search_internal_result_num, self.metric
                )
            else:  # KDT
                nearest_centroids = self.tree.search(
                    query, self.centroids, search_internal_result_num, self.metric
                )
        else:
            if self.metric == 'L2':
                centroid_dists = np.sum((self.centroids - query) ** 2, axis=1)
            elif self.metric in ('IP', 'Cosine'):
                centroid_dists = -np.dot(self.centroids, query)
            nearest_centroids = np.argsort(centroid_dists)[:search_internal_result_num]
        
        # Batch load postings
        postings = self._load_postings_batch(list(nearest_centroids))
        
        # Parallel search postings
        if self.num_threads > 1:
            results = self._search_postings_parallel(query, postings, nearest_centroids, max_check)
        else:
            results = self._search_postings_sequential(query, postings, nearest_centroids, max_check)
        
        all_indices, all_dists = results
        
        if len(all_indices) == 0:
            return np.array([]), np.array([])
        
        all_indices = np.array(all_indices)
        
        if self.use_rabitq:
            if self.metric == 'L2':
                true_dists = np.sum((data[all_indices] - query) ** 2, axis=1)
            elif self.metric == 'IP':
                true_dists = -np.dot(data[all_indices], query)
            elif self.metric == 'Cosine':
                true_dists = -np.dot(data[all_indices], query)
            top_k_idx = np.argsort(true_dists)[:k]
        else:
            true_dists = np.array(all_dists)
            top_k_idx = np.argsort(true_dists)[:k]
        
        return true_dists[top_k_idx], all_indices[top_k_idx]
    
    def _search_postings_sequential(self, query, postings, nearest_centroids, max_check):
        """Sequential posting search (original)"""
        seen = set()
        all_indices = []
        all_dists = []
        
        for centroid_id in nearest_centroids:
            if centroid_id not in postings:
                continue
            
            posting_ids, codes, rabitq = postings[centroid_id]
            search_k = min(max_check, len(posting_ids))
            
            if self.use_rabitq:
                _, local_indices = rabitq.search(query, codes, k=search_k)
                local_dists = None
            else:
                if self.metric == 'L2':
                    dists = np.sum((codes - query) ** 2, axis=1)
                elif self.metric == 'IP':
                    dists = -np.dot(codes, query)
                elif self.metric == 'Cosine':
                    dists = -np.dot(codes, query)
                local_indices = np.argsort(dists)[:search_k]
                local_dists = dists[local_indices]
            
            for idx, local_idx in enumerate(local_indices):
                global_id = posting_ids[local_idx]
                if global_id not in seen:
                    seen.add(global_id)
                    all_indices.append(global_id)
                    if local_dists is not None:
                        all_dists.append(local_dists[idx])
                    if len(all_indices) >= max_check:
                        break
            
            if len(all_indices) >= max_check:
                break
        
        return all_indices, all_dists
    
    def _search_one_posting(self, args):
        """Search one posting (for parallel execution)"""
        centroid_id, query, postings, max_check = args
        
        if centroid_id not in postings:
            return []
        
        posting_ids, codes, rabitq = postings[centroid_id]
        search_k = min(max_check, len(posting_ids))
        
        results = []
        
        if self.use_rabitq:
            _, local_indices = rabitq.search(query, codes, k=search_k)
            for local_idx in local_indices:
                results.append((posting_ids[local_idx], None))
        else:
            if self.metric == 'L2':
                dists = np.sum((codes - query) ** 2, axis=1)
            elif self.metric == 'IP':
                dists = -np.dot(codes, query)
            elif self.metric == 'Cosine':
                dists = -np.dot(codes, query)
            local_indices = np.argsort(dists)[:search_k]
            local_dists = dists[local_indices]
            for idx, local_idx in enumerate(local_indices):
                results.append((posting_ids[local_idx], local_dists[idx]))
        
        return results
    
    def _search_postings_parallel(self, query, postings, nearest_centroids, max_check):
        """Parallel posting search using ThreadPoolExecutor"""
        # Note: Use ProcessPoolExecutor if Numba parallel causes issues
        # For now, process in chunks to reduce thread conflicts
        
        chunk_size = max(1, len(nearest_centroids) // self.num_threads)
        chunks = [nearest_centroids[i:i+chunk_size] for i in range(0, len(nearest_centroids), chunk_size)]
        
        def search_chunk(centroid_ids):
            results = []
            for cid in centroid_ids:
                if cid not in postings:
                    continue
                posting_ids, codes, rabitq = postings[cid]
                search_k = min(max_check, len(posting_ids))
                
                if self.use_rabitq:
                    _, local_indices = rabitq.search(query, codes, k=search_k)
                    for local_idx in local_indices:
                        results.append((posting_ids[local_idx], None))
                else:
                    if self.metric == 'L2':
                        dists = np.sum((codes - query) ** 2, axis=1)
                    elif self.metric == 'IP':
                        dists = -np.dot(codes, query)
                    elif self.metric == 'Cosine':
                        dists = -np.dot(codes, query)
                    local_indices = np.argsort(dists)[:search_k]
                    local_dists = dists[local_indices]
                    for idx, local_idx in enumerate(local_indices):
                        results.append((posting_ids[local_idx], local_dists[idx]))
            return results
        
        # Execute chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results_list = list(executor.map(search_chunk, chunks))
        
        # Merge results and deduplicate
        seen = set()
        all_indices = []
        all_dists = []
        
        for results in results_list:
            for global_id, dist in results:
                if global_id not in seen:
                    seen.add(global_id)
                    all_indices.append(global_id)
                    if dist is not None:
                        all_dists.append(dist)
                    if len(all_indices) >= max_check:
                        break
            if len(all_indices) >= max_check:
                break
        
        return all_indices, all_dists
    
    def print_cache_stats(self):
        """Print cache statistics"""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = self._cache_hits / total * 100
            print(f"Cache: {self._cache_hits} hits, {self._cache_misses} misses ({hit_rate:.1f}% hit rate)")
