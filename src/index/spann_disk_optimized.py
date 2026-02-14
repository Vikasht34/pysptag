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
from numba import njit
from ..core.rng import RNG, MetricType
from ..quantization.rabitq_numba import RaBitQNumba
from ..clustering import ClusteringAlgorithm, KMeansClustering, HierarchicalClustering
from ..core.sptag_params import (
    DEFAULT_REPLICA_COUNT,
    DEFAULT_POSTING_VECTOR_LIMIT,
    DEFAULT_POSTING_PAGE_LIMIT,
    DEFAULT_INTERNAL_RESULT_NUM,
    DEFAULT_RNG_FACTOR,
    get_sptag_posting_limit
)


@njit
def deduplicate_with_hash(candidates, hash_size=8192):
    """Fast deduplication with fixed-size hash table"""
    hash_table = np.full(hash_size, -1, dtype=np.int32)
    unique = []
    
    for candidate in candidates:
        idx = candidate % hash_size
        # Linear probing
        probes = 0
        while hash_table[idx] != -1 and hash_table[idx] != candidate and probes < hash_size:
            idx = (idx + 1) % hash_size
            probes += 1
        
        if hash_table[idx] == -1:
            hash_table[idx] = candidate
            unique.append(candidate)
    
    return np.array(unique, dtype=np.int32)


class SPANNDiskOptimized:
    """Optimized disk-based SPANN with pluggable clustering"""
    
    def __init__(
        self,
        dim: int,
        target_posting_size: int = None,  # Auto-calculate from SPTAG defaults
        replica_count: int = DEFAULT_REPLICA_COUNT,  # 8
        bq: int = 4,
        use_rabitq: bool = True,
        metric: MetricType = 'L2',
        tree_type: str = 'BKT',  # BKT or KDT
        disk_path: str = './spann_index',
        cache_size: int = 128,
        num_threads: int = 1,
        clustering: str = 'hierarchical',
        use_rng_filtering: bool = True,
        preload_postings: bool = False,
        use_faiss_centroids: bool = True,
        # SPTAG-exact parameters
        posting_vector_limit: int = DEFAULT_POSTING_VECTOR_LIMIT,  # 118
        posting_page_limit: int = DEFAULT_POSTING_PAGE_LIMIT,  # 3
        internal_result_num: int = DEFAULT_INTERNAL_RESULT_NUM,  # 64
        rng_factor: float = DEFAULT_RNG_FACTOR,  # 1.0
        centroid_ratio: float = 0.01  # Ratio of centroids to data (1% = 10K centroids for 1M vectors)
    ):
        self.dim = dim
        self.replica_count = replica_count
        self.bq = bq
        self.use_rabitq = use_rabitq
        self.metric = metric
        self.tree_type = tree_type
        self.disk_path = disk_path
        self.cache_size = cache_size
        self.num_threads = num_threads
        self.use_rng_filtering = use_rng_filtering
        self.preload_postings = preload_postings
        self.use_faiss_centroids = use_faiss_centroids
        self.centroid_ratio = centroid_ratio
        
        # SPTAG-exact parameters
        self.posting_vector_limit = posting_vector_limit
        self.posting_page_limit = posting_page_limit
        self.internal_result_num = internal_result_num
        self.rng_factor = rng_factor
        
        # Calculate SPTAG-exact posting size limit
        if target_posting_size is None:
            # Use SPTAG formula
            value_size = 4 if not use_rabitq else ((dim + 3) // 4) / dim
            self.target_posting_size = int(get_sptag_posting_limit(
                dim, value_size, posting_vector_limit, posting_page_limit
            ))
            print(f"Auto-calculated posting size: {self.target_posting_size} vectors (SPTAG-exact)")
        else:
            self.target_posting_size = int(target_posting_size)
        
        # Select clustering algorithm
        if clustering == 'hierarchical':
            self.clusterer = HierarchicalClustering(
                select_threshold=0,  # Auto-compute from ratio (SPTAG default)
                split_threshold=0,  # Auto-compute from ratio (SPTAG default)
                split_factor=0,  # Auto-compute from ratio (SPTAG default)
                ratio=centroid_ratio,  # Use configurable ratio
                kmeans_k=32,
                leaf_size=8,
                metric=metric
            )
        else:  # kmeans
            self.clusterer = KMeansClustering(metric=metric)
        
        # Create tree (always create, even if using faiss for search)
        if tree_type == 'BKT':
            from ..core.bktree import BKTree
            self.tree = BKTree(num_trees=1, kmeans_k=32)
        else:  # KDT
            from ..core.kdtree import KDTree
            self.tree = KDTree(num_trees=1)
        
        self._centroid_index = None  # Will be initialized after build if use_faiss_centroids
            
        self.rng = RNG(neighborhood_size=32, metric=metric)
        
        # Cache for posting lists
        self._posting_cache: Dict[int, Tuple] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._bytes_read = 0  # Track bytes read per query
        
        # Shared RaBitQ instance (avoid JIT recompilation)
        if use_rabitq:
            self._shared_rabitq = RaBitQNumba(dim=dim, bq=bq, metric=metric)
            # Trigger JIT compilation immediately with dummy data
            dummy_query = np.zeros(dim, dtype=np.float32)
            dummy_codes = np.zeros((10, (dim + 3) // 4), dtype=np.uint8)
            self._shared_rabitq.centroid = np.zeros(dim, dtype=np.float32)
            self._shared_rabitq.scale = np.ones(dim, dtype=np.float32)
            self._shared_rabitq.res_min = np.zeros(dim, dtype=np.float32)
            try:
                _ = self._shared_rabitq.search(dummy_query, dummy_codes, k=5)
            except:
                pass  # Ignore errors, just want to trigger JIT
        else:
            self._shared_rabitq = None
        
        # Create disk directory
        os.makedirs(disk_path, exist_ok=True)
        os.makedirs(os.path.join(disk_path, 'postings'), exist_ok=True)
    
    def preload_all_postings(self):
        """Preload all postings into memory for faster search."""
        print(f"Preloading {self.num_clusters} postings into memory...")
        for i in range(self.num_clusters):
            self._load_posting_mmap(i)
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i+1}/{self.num_clusters} postings...")
        print(f"✓ All postings preloaded ({len(self._posting_cache)} in cache)")
    
    def build(self, data: np.ndarray):
        """Build index with SPTAG-style clustering"""
        n, dim = data.shape
        print(f"Building optimized SPANN for {n} vectors")
        print(f"  Clustering: {self.clusterer.__class__.__name__}")
        print(f"  Posting limit: {self.target_posting_size} vectors")
        print(f"  Replica count: {self.replica_count}")
        
        # Step 1: Cluster data (use ratio-based for SPTAG style)
        print("[1/5] Clustering...")
        self.centroids, labels = self.clusterer.cluster(data, target_clusters=None)  # None = use ratio
        self.num_clusters = len(self.centroids)
        print(f"  Created {self.num_clusters} clusters ({self.num_clusters/n*100:.2f}% of data)")
        
        # Step 2: Assign with replicas and posting limits
        print(f"[2/5] Assigning vectors to {self.num_clusters} centroids...")
        self.posting_lists, replica_counts = self.clusterer.assign_with_replicas(
            data, self.centroids, self.replica_count, self.target_posting_size,
            use_rng_filtering=self.use_rng_filtering
        )
        
        # Step 3: Save posting lists in single file format (SPTAG-style)
        print("[3/5] Saving posting lists in single file format...")
        total_original = 0
        total_compressed = 0
        
        # First pass: serialize all postings and collect offsets
        single_file = os.path.join(self.disk_path, 'postings.bin')
        offset_table = []
        serialized_postings = []
        
        for i, posting_ids in enumerate(self.posting_lists):
            if len(posting_ids) == 0:
                offset_table.append((i, 0, 0))
                serialized_postings.append(b'')
                continue
            
            posting_ids = np.array(posting_ids, dtype=np.int32)
            posting_vecs = data[posting_ids]
            
            if self.use_rabitq:
                rabitq = RaBitQNumba(dim=self.dim, bq=self.bq, metric=self.metric)
                codes = rabitq.build(posting_vecs)
                posting_bytes = self._serialize_posting(posting_ids, codes, rabitq)
                total_original += posting_vecs.nbytes
                total_compressed += len(posting_bytes)
            else:
                posting_bytes = self._serialize_posting(posting_ids, posting_vecs, None)
                total_original += posting_vecs.nbytes
                total_compressed += len(posting_bytes)
            
            serialized_postings.append(posting_bytes)
            offset_table.append((i, 0, len(posting_bytes)))  # offset filled later
            
            if (i + 1) % 1000 == 0:
                print(f"  Serialized {i+1}/{self.num_clusters} postings")
        
        # Second pass: write file with correct offsets
        print(f"  Writing single file...")
        with open(single_file, 'wb') as f:
            # Write header
            header_size = 12
            offset_table_size = self.num_clusters * 20
            f.write(struct.pack('I', self.num_clusters))
            f.write(struct.pack('Q', offset_table_size))
            
            # Calculate offsets
            data_start = header_size + offset_table_size
            current_offset = data_start
            for i in range(len(offset_table)):
                cid, _, size = offset_table[i]
                offset_table[i] = (cid, current_offset, size)
                current_offset += size
            
            # Write offset table
            for cid, offset, size in offset_table:
                f.write(struct.pack('I', cid))
                f.write(struct.pack('Q', offset))
                f.write(struct.pack('Q', size))
            
            # Write posting data
            for posting_bytes in serialized_postings:
                if len(posting_bytes) > 0:
                    f.write(posting_bytes)
        
        if self.use_rabitq:
            print(f"  Original: {total_original/1024**2:.2f} MB")
            print(f"  Compressed: {total_compressed/1024**2:.2f} MB")
            print(f"  Savings: {(1 - total_compressed/total_original)*100:.1f}%")
        else:
            print(f"  Stored: {total_original/1024**2:.2f} MB (no compression)")
        
        # Step 4: Build tree + RNG
        print(f"[4/5] Building {self.tree_type}+RNG on centroids...")
        self.tree.build(self.centroids)
        
        # Build initial graph from tree search
        print(f"  Building initial RNG graph using k-NN...")
        import faiss
        # Use faiss for fast k-NN (metric-aware)
        if self.metric == 'L2':
            index_knn = faiss.IndexFlatL2(self.centroids.shape[1])
        else:  # IP or Cosine
            index_knn = faiss.IndexFlatIP(self.centroids.shape[1])
        index_knn.add(self.centroids.astype(np.float32))
        
        init_graph = []
        batch_size = 1000
        for i in range(0, len(self.centroids), batch_size):
            end = min(i + batch_size, len(self.centroids))
            batch = self.centroids[i:end].astype(np.float32)
            # Search for k+1 neighbors (includes self)
            _, neighbors = index_knn.search(batch, self.rng.neighborhood_size + 1)
            # Remove self from neighbors
            for j, neighb in enumerate(neighbors):
                # Filter out self (distance 0)
                filtered = neighb[neighb != (i + j)][:self.rng.neighborhood_size]
                init_graph.append(filtered)
            
            if (i + batch_size) % 5000 == 0:
                print(f"    Processed {min(i + batch_size, len(self.centroids))}/{len(self.centroids)} centroids")
        
        self.rng.build(self.centroids, init_graph=init_graph)
        
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
            'rng': self.rng,
            'use_faiss_centroids': self.use_faiss_centroids
        }
        with open(os.path.join(self.disk_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Build faiss index for fast centroid search (metric-aware)
        if self.use_faiss_centroids:
            print("  Building faiss centroid index...")
            import faiss
            if self.metric == 'L2':
                self._centroid_index = faiss.IndexFlatL2(self.dim)
            else:  # IP or Cosine
                self._centroid_index = faiss.IndexFlatIP(self.dim)
            self._centroid_index.add(self.centroids.astype(np.float32))
            print(f"  ✓ Faiss index ready ({len(self.centroids)} centroids, metric={self.metric})")
        
        print(f"✓ Index built and saved to {self.disk_path}")
    
    def _serialize_posting(self, posting_ids: np.ndarray, codes: np.ndarray, 
                          rabitq: Optional[RaBitQNumba]) -> bytes:
        """Serialize posting to bytes"""
        import io
        buf = io.BytesIO()
        
        # Header
        num_vecs = len(posting_ids)
        code_dim = codes.shape[1] if len(codes.shape) > 1 else codes.shape[0]
        is_unquantized = 1 if rabitq is None else 0
        buf.write(struct.pack('III', num_vecs, code_dim, is_unquantized))
        
        # Data
        buf.write(posting_ids.tobytes())
        buf.write(codes.tobytes())
        
        # RaBitQ params (if quantized)
        if rabitq is not None:
            import pickle
            rabitq_bytes = pickle.dumps(rabitq)
            buf.write(struct.pack('I', len(rabitq_bytes)))
            buf.write(rabitq_bytes)
        
        return buf.getvalue()
    
    def _load_posting_mmap(self, centroid_id: int, max_vectors: int = None):
        """Load posting from single file with memory-mapped I/O
        
        Args:
            centroid_id: Cluster ID
            max_vectors: Maximum vectors to load (None = load all)
        """
        # Check cache first
        if centroid_id in self._posting_cache:
            self._cache_hits += 1
            cached = self._posting_cache[centroid_id]
            if max_vectors is None:
                return cached
            # Return limited version
            return (cached[0][:max_vectors] if cached[0] is not None else None, 
                    cached[1], cached[2])
        
        self._cache_misses += 1
        
        # Load offset table if not loaded
        if not hasattr(self, '_offset_table'):
            self._load_offset_table()
        
        # Get offset and size
        if centroid_id >= len(self._offset_table):
            return None, None, None
        
        offset, size = self._offset_table[centroid_id]
        if size == 0:  # Empty posting
            return None, None, None
        
        # Memory-map the single file
        single_file = os.path.join(self.disk_path, 'postings.bin')
        with open(single_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read header
            num_vecs_total, code_dim, is_unquantized = struct.unpack('III', mm[offset:offset+12])
            
            # Limit vectors to load (but read RaBitQ from full posting)
            num_vecs_to_load = num_vecs_total
            if max_vectors is not None:
                num_vecs_to_load = min(num_vecs_total, max_vectors)
            
            # Track bytes read
            bytes_to_read = 12 + num_vecs_to_load * 4
            if is_unquantized:
                bytes_to_read += num_vecs_to_load * code_dim * 4
            else:
                bytes_to_read += num_vecs_to_load * code_dim
            self._bytes_read += bytes_to_read
            
            # Read posting IDs (only what we need)
            pos = offset + 12
            posting_ids = np.frombuffer(mm, dtype=np.int32, count=num_vecs_to_load, offset=pos)
            pos_ids_end = offset + 12 + num_vecs_total * 4  # Full IDs section
            
            # Read codes (only what we need)
            if is_unquantized:
                codes = np.frombuffer(mm, dtype=np.float32, count=num_vecs_to_load * code_dim, offset=pos_ids_end)
                codes = codes.reshape(num_vecs_to_load, code_dim)
                rabitq = None
            else:
                codes = np.frombuffer(mm, dtype=np.uint8, count=num_vecs_to_load * code_dim, offset=pos_ids_end)
                codes = codes.reshape(num_vecs_to_load, code_dim)
                
                # Read RaBitQ params from AFTER full codes section
                pos_rabitq = pos_ids_end + num_vecs_total * code_dim
                rabitq_size = struct.unpack('I', mm[pos_rabitq:pos_rabitq+4])[0]
                pos_rabitq += 4
                import pickle
                rabitq_params = pickle.loads(mm[pos_rabitq:pos_rabitq+rabitq_size])
                
                # Copy params to shared instance
                if self._shared_rabitq is not None:
                    self._shared_rabitq.centroid = rabitq_params.centroid
                    self._shared_rabitq.scale = rabitq_params.scale
                    self._shared_rabitq.res_min = rabitq_params.res_min
                    rabitq = self._shared_rabitq
                else:
                    rabitq = rabitq_params
        
        # Cache result
        result = (posting_ids, codes, rabitq)
        if len(self._posting_cache) < self.cache_size:
            self._posting_cache[centroid_id] = result
        
        return result
    
    def _load_offset_table(self):
        """Load offset table from single file"""
        single_file = os.path.join(self.disk_path, 'postings.bin')
        with open(single_file, 'rb') as f:
            # Read header
            num_postings = struct.unpack('I', f.read(4))[0]
            offset_table_size = struct.unpack('Q', f.read(8))[0]
            
            # Read offset table
            self._offset_table = []
            for _ in range(num_postings):
                data = f.read(20)
                if len(data) < 20:
                    break
                cid = struct.unpack('I', data[0:4])[0]
                offset = struct.unpack('Q', data[4:12])[0]
                size = struct.unpack('Q', data[12:20])[0]
                self._offset_table.append((offset, size))
    
    
    def _load_postings_batch(self, centroid_ids: list, max_vectors_per_posting: int = None):
        """Batch load multiple postings with vector limit (SPTAG-style)
        
        Args:
            centroid_ids: List of cluster IDs
            max_vectors_per_posting: Max vectors to load per posting (None = load all)
        """
        postings = {}
        uncached = []
        
        # Check cache first
        for cid in centroid_ids:
            if cid in self._posting_cache:
                cached = self._posting_cache[cid]
                # Apply limit to cached results
                postings[cid] = (cached[0][:max_vectors_per_posting] if cached[0] is not None else None,
                                cached[1], cached[2])
                self._cache_hits += 1
            else:
                uncached.append(cid)
        
        # Load uncached in batch with limit
        for cid in uncached:
            result = self._load_posting_mmap(cid, max_vectors=max_vectors_per_posting)
            if result[0] is not None:
                postings[cid] = result
                # Cache the result
                if len(self._posting_cache) < self.cache_size:
                    self._posting_cache[cid] = result
                self._cache_misses += 1
        
        return postings
    
    def _search_with_async_pruning(
        self, query: np.ndarray, data: np.ndarray, 
        centroid_ids: np.ndarray, k: int,
        search_internal_result_num: int, max_check: int,
        max_vectors_per_posting: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SPTAG-style async batch I/O with deterministic processing
        
        Submits all I/O requests at once, then processes in centroid distance order
        to ensure deterministic results (not completion order).
        """
        from concurrent.futures import ThreadPoolExecutor
        
        all_indices = []
        all_dists = []
        seen = set()
        loaded_count = 0
        
        # Submit all I/O requests at once (SPTAG's BatchReadFileAsync)
        with ThreadPoolExecutor(max_workers=min(8, len(centroid_ids))) as executor:
            # Submit all loads - map centroid_id to future
            futures = {
                cid: executor.submit(self._load_posting_mmap, cid, max_vectors_per_posting)
                for cid in centroid_ids
            }
            
            # Process in centroid distance order (deterministic)
            for cid in centroid_ids:
                result = futures[cid].result()  # Wait for this specific centroid
                if result[0] is None:
                    continue
                
                posting_ids, codes, rabitq = result
                loaded_count += 1
                n_vectors = len(posting_ids)
                search_k = min(max_check, n_vectors)
                
                # Search posting
                if self.use_rabitq and rabitq is not None:
                    _, local_indices = rabitq.search(query, codes, k=search_k)
                    # Safety: clip indices to actual posting size
                    local_indices = local_indices[local_indices < n_vectors]
                else:
                    if self.metric == 'L2':
                        dists = np.sum((codes - query) ** 2, axis=1)
                        local_indices = np.argsort(dists)[:search_k]
                    else:  # IP or Cosine - higher is better
                        dists = np.dot(codes, query)
                        if search_k >= n_vectors:
                            local_indices = np.argsort(-dists)
                        else:
                            local_indices = np.argpartition(dists, -search_k)[-search_k:]
                            local_indices = local_indices[np.argsort(-dists[local_indices])]
                
                # Collect results with deduplication and bounds checking
                for local_idx in local_indices:
                    global_id = posting_ids[local_idx]
                    if global_id not in seen and global_id < len(data):
                        seen.add(global_id)
                        all_indices.append(global_id)
                    if len(all_indices) >= max_check:
                        break
                
                # Early termination (SPTAG's query-aware pruning)
                if loaded_count >= search_internal_result_num and len(all_indices) >= k * 2:
                    break
        
        all_indices = np.array(all_indices) if all_indices else np.array([])
        
        if len(all_indices) == 0:
            return np.array([]), np.array([])
        
        # Rerank with actual distances
        if self.use_rabitq:
            if self.metric == 'L2':
                all_dists = np.sum((data[all_indices] - query) ** 2, axis=1)
            elif self.metric in ('IP', 'Cosine'):
                all_dists = -np.dot(data[all_indices], query)  # Negate for sorting
        else:
            # Without RaBitQ, still need to compute distances for reranking!
            if self.metric == 'L2':
                all_dists = np.sum((data[all_indices] - query) ** 2, axis=1)
            elif self.metric in ('IP', 'Cosine'):
                all_dists = -np.dot(data[all_indices], query)  # Negate for sorting
        
        # Get top-k (smallest distances = best for L2, largest for IP after negation)
        if len(all_indices) > k:
            top_k_idx = np.argpartition(all_dists, k-1)[:k]
            top_k_idx = top_k_idx[np.argsort(all_dists[top_k_idx])]
            return all_indices[top_k_idx], all_dists[top_k_idx]
        else:
            sorted_idx = np.argsort(all_dists)
            return all_indices[sorted_idx], all_dists[sorted_idx]
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
        search_internal_result_num: int = 32,
        max_check: int = 4096,
        max_dist_ratio: float = 10000.0,
        use_async_pruning: bool = False,  # Enable async query-aware pruning
        max_vectors_per_posting: int = None  # NEW: Limit vectors per posting
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized search with faiss centroids + RaBitQ"""
        
        # Find nearest centroids - use faiss if enabled
        if self.use_faiss_centroids and self._centroid_index is not None:
            # Fast faiss search (~0.5ms for 11K centroids)
            centroid_dists, nearest_centroids = self._centroid_index.search(
                query.reshape(1, -1).astype(np.float32), 
                search_internal_result_num
            )
            centroid_dists = centroid_dists[0]
            nearest_centroids = nearest_centroids[0]
        elif hasattr(self, 'rng') and len(self.rng.graph) > 0:
            if self.tree_type == 'BKT':
                from ..core.bktree_rng_search import bktree_rng_search
                nearest_centroids = bktree_rng_search(
                    query, self.centroids, self.tree.tree_roots,
                    self.tree.tree_start, self.rng.graph,
                    search_internal_result_num, self.metric
                )
                # Compute distances for filtering
                if self.metric == 'L2':
                    centroid_dists = np.sum((self.centroids[nearest_centroids] - query) ** 2, axis=1)
                elif self.metric in ('IP', 'Cosine'):
                    centroid_dists = -np.dot(self.centroids[nearest_centroids], query)
            else:  # KDT
                nearest_centroids = self.tree.search(
                    query, self.centroids, search_internal_result_num, self.metric
                )
                # Compute distances for filtering
                if self.metric == 'L2':
                    centroid_dists = np.sum((self.centroids[nearest_centroids] - query) ** 2, axis=1)
                elif self.metric in ('IP', 'Cosine'):
                    centroid_dists = -np.dot(self.centroids[nearest_centroids], query)
        else:
            if self.metric == 'L2':
                centroid_dists = np.sum((self.centroids - query) ** 2, axis=1)
            elif self.metric in ('IP', 'Cosine'):
                centroid_dists = -np.dot(self.centroids, query)
            sorted_idx = np.argsort(centroid_dists)
            nearest_centroids = sorted_idx[:search_internal_result_num]
            centroid_dists = centroid_dists[sorted_idx[:search_internal_result_num]]
        
        # SPTAG-style distance filtering: limitDist = first_dist * maxDistRatio
        limit_dist = centroid_dists[0] * max_dist_ratio
        
        # Filter centroids: stop if distance > limitDist (and limitDist > 0.1)
        valid_count = search_internal_result_num
        if limit_dist > 0.1:
            for i in range(len(centroid_dists)):
                if centroid_dists[i] > limit_dist:
                    valid_count = i
                    break
        
        # Use only valid centroids
        nearest_centroids = nearest_centroids[:valid_count]
        
        # Choose loading strategy
        if use_async_pruning:
            # SPTAG-style: Async batch I/O with query-aware pruning
            return self._search_with_async_pruning(
                query, data, nearest_centroids, k, 
                search_internal_result_num, max_check, max_vectors_per_posting
            )
        else:
            # Current: Batch load all postings
            postings = self._load_postings_batch(list(nearest_centroids), max_vectors_per_posting)
            
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
        
        return all_indices[top_k_idx], true_dists[top_k_idx]
    
    def search_batch(
        self,
        queries: np.ndarray,
        data: np.ndarray,
        k: int = 10,
        search_internal_result_num: int = 32,
        max_check: int = 4096,
        max_dist_ratio: float = 10000.0,
        use_async_pruning: bool = False,
        max_vectors_per_posting: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch search multiple queries with vectorized centroid search"""
        batch_size = len(queries)
        all_ids = np.zeros((batch_size, k), dtype=np.int32)
        all_dists = np.zeros((batch_size, k), dtype=np.float32)
        
        # Vectorized centroid search for all queries at once
        if self.use_faiss_centroids and self._centroid_index is not None:
            centroid_dists_batch, nearest_centroids_batch = self._centroid_index.search(
                queries.astype(np.float32), search_internal_result_num
            )
        else:
            # Vectorized distance computation
            if self.metric == 'L2':
                centroid_dists_batch = np.sum((queries[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
            elif self.metric in ('IP', 'Cosine'):
                centroid_dists_batch = -np.dot(queries, self.centroids.T)
            nearest_centroids_batch = np.argsort(centroid_dists_batch, axis=1)[:, :search_internal_result_num]
            centroid_dists_batch = np.take_along_axis(centroid_dists_batch, nearest_centroids_batch, axis=1)
        
        # Search each query (can't fully batch posting list search)
        for i in range(batch_size):
            ids, dists = self.search(
                queries[i], data, k=k,
                search_internal_result_num=search_internal_result_num,
                max_check=max_check,
                max_dist_ratio=max_dist_ratio,
                use_async_pruning=use_async_pruning,
                max_vectors_per_posting=max_vectors_per_posting
            )
            all_ids[i] = ids
            all_dists[i] = dists
        
        return all_ids, all_dists
    
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
                    local_indices = np.argsort(dists)[:search_k]
                elif self.metric in ('IP', 'Cosine'):
                    dists = np.dot(codes, query)
                    local_indices = np.argpartition(dists, -search_k)[-search_k:]
                    local_indices = local_indices[np.argsort(-dists[local_indices])]
                    dists = -dists  # Negate for consistent distance semantics
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
                local_indices = np.argsort(dists)[:search_k]
            elif self.metric in ('IP', 'Cosine'):
                dists = np.dot(codes, query)
                local_indices = np.argpartition(dists, -search_k)[-search_k:]
                local_indices = local_indices[np.argsort(-dists[local_indices])]
                dists = -dists  # Negate for consistent distance semantics
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
                        local_indices = np.argsort(dists)[:search_k]
                    elif self.metric in ('IP', 'Cosine'):
                        dists = np.dot(codes, query)
                        local_indices = np.argpartition(dists, -search_k)[-search_k:]
                        local_indices = local_indices[np.argsort(-dists[local_indices])]
                        dists = -dists  # Negate for consistent distance semantics
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
