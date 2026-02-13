"""
Detailed latency profiling for disk-based SPANN
Breaks down every component of search latency
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

class ProfiledSPANN(SPANNDiskOptimized):
    """SPANN with detailed profiling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_stats()
    
    def reset_stats(self):
        """Reset profiling statistics"""
        self.stats = {
            'centroid_search_time': [],
            'posting_load_time': [],
            'posting_search_time': [],
            'rerank_time': [],
            'num_postings_touched': [],
            'num_vectors_in_postings': [],
            'num_candidates': [],
            'posting_sizes_bytes': [],
            'total_data_loaded_mb': [],
        }
    
    def search(self, query, data, k=10, search_internal_result_num=64, max_check=4096):
        """Search with detailed profiling"""
        t_total = time.time()
        
        # 1. Find nearest centroids
        t0 = time.time()
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
        
        centroid_search_time = (time.time() - t0) * 1000
        self.stats['centroid_search_time'].append(centroid_search_time)
        
        # 2. Batch load postings
        t0 = time.time()
        postings = self._load_postings_batch(list(nearest_centroids))
        posting_load_time = (time.time() - t0) * 1000
        self.stats['posting_load_time'].append(posting_load_time)
        
        # Track posting statistics
        num_postings = len(postings)
        total_vectors = sum(len(p[0]) for p in postings.values())
        total_bytes = 0
        for cid, (posting_ids, codes, rabitq) in postings.items():
            total_bytes += posting_ids.nbytes + codes.nbytes
        
        self.stats['num_postings_touched'].append(num_postings)
        self.stats['num_vectors_in_postings'].append(total_vectors)
        self.stats['posting_sizes_bytes'].append(total_bytes)
        self.stats['total_data_loaded_mb'].append(total_bytes / 1024 / 1024)
        
        # 3. Search postings
        t0 = time.time()
        results = self._search_postings_sequential(query, postings, nearest_centroids, max_check)
        posting_search_time = (time.time() - t0) * 1000
        self.stats['posting_search_time'].append(posting_search_time)
        
        all_indices, all_dists = results
        self.stats['num_candidates'].append(len(all_indices))
        
        if len(all_indices) == 0:
            return np.array([]), np.array([])
        
        all_indices = np.array(all_indices)
        
        # 4. Rerank with true distances
        t0 = time.time()
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
        
        rerank_time = (time.time() - t0) * 1000
        self.stats['rerank_time'].append(rerank_time)
        
        return true_dists[top_k_idx], all_indices[top_k_idx]
    
    def print_detailed_stats(self):
        """Print detailed profiling statistics"""
        print("\n" + "="*80)
        print("DETAILED LATENCY BREAKDOWN")
        print("="*80)
        
        def print_stat(name, values, unit='ms'):
            if not values:
                return
            avg = np.mean(values)
            p50 = np.percentile(values, 50)
            p90 = np.percentile(values, 90)
            p99 = np.percentile(values, 99)
            total = np.sum(values)
            print(f"{name:<30} avg={avg:>7.2f}{unit}  p50={p50:>7.2f}{unit}  p90={p90:>7.2f}{unit}  p99={p99:>7.2f}{unit}")
        
        print("\nTiming Breakdown:")
        print("-" * 80)
        print_stat("1. Centroid Search", self.stats['centroid_search_time'])
        print_stat("2. Posting Load (disk I/O)", self.stats['posting_load_time'])
        print_stat("3. Posting Search (quantized)", self.stats['posting_search_time'])
        print_stat("4. Reranking (true distances)", self.stats['rerank_time'])
        
        total_time = [
            self.stats['centroid_search_time'][i] +
            self.stats['posting_load_time'][i] +
            self.stats['posting_search_time'][i] +
            self.stats['rerank_time'][i]
            for i in range(len(self.stats['centroid_search_time']))
        ]
        print_stat("TOTAL", total_time)
        
        print("\nData Statistics:")
        print("-" * 80)
        print(f"{'Postings touched':<30} avg={np.mean(self.stats['num_postings_touched']):>7.1f}  "
              f"p50={np.percentile(self.stats['num_postings_touched'], 50):>7.1f}  "
              f"p90={np.percentile(self.stats['num_postings_touched'], 90):>7.1f}")
        
        print(f"{'Vectors in postings':<30} avg={np.mean(self.stats['num_vectors_in_postings']):>7.0f}  "
              f"p50={np.percentile(self.stats['num_vectors_in_postings'], 50):>7.0f}  "
              f"p90={np.percentile(self.stats['num_vectors_in_postings'], 90):>7.0f}")
        
        print(f"{'Candidates for reranking':<30} avg={np.mean(self.stats['num_candidates']):>7.0f}  "
              f"p50={np.percentile(self.stats['num_candidates'], 50):>7.0f}  "
              f"p90={np.percentile(self.stats['num_candidates'], 90):>7.0f}")
        
        print(f"{'Data loaded per query (MB)':<30} avg={np.mean(self.stats['total_data_loaded_mb']):>7.2f}  "
              f"p50={np.percentile(self.stats['total_data_loaded_mb'], 50):>7.2f}  "
              f"p90={np.percentile(self.stats['total_data_loaded_mb'], 90):>7.2f}")
        
        print("\nPercentage Breakdown:")
        print("-" * 80)
        avg_total = np.mean(total_time)
        pct_centroid = np.mean(self.stats['centroid_search_time']) / avg_total * 100
        pct_load = np.mean(self.stats['posting_load_time']) / avg_total * 100
        pct_search = np.mean(self.stats['posting_search_time']) / avg_total * 100
        pct_rerank = np.mean(self.stats['rerank_time']) / avg_total * 100
        
        print(f"Centroid Search:     {pct_centroid:>6.1f}%")
        print(f"Posting Load (I/O):  {pct_load:>6.1f}%")
        print(f"Posting Search:      {pct_search:>6.1f}%")
        print(f"Reranking:           {pct_rerank:>6.1f}%")
        print("="*80)
        
        # Cache stats
        self.print_cache_stats()


# Test with profiling
print("="*80)
print("Detailed Latency Profiling")
print("="*80)

# Generate test data (10K vectors, 768-dim like Cohere)
np.random.seed(42)
data = np.random.randn(10000, 768).astype(np.float32)
queries = np.random.randn(100, 768).astype(np.float32)

print("\nBuilding index...")
index = ProfiledSPANN(
    dim=768,
    target_posting_size=500,
    replica_count=6,
    bq=4,
    use_rabitq=True,
    tree_type='KDT',
    disk_path='./profile_index',
    cache_size=128,
    num_threads=1
)
index.build(data)

print("\nSearching (first pass - cold cache)...")
index.reset_stats()
for i, query in enumerate(queries[:20]):  # First 20 queries
    dists, indices = index.search(query, data, k=10, search_internal_result_num=64)
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/20 queries")

print("\n" + "="*80)
print("COLD CACHE RESULTS (first 20 queries)")
index.print_detailed_stats()

print("\n\nSearching (second pass - warm cache)...")
index.reset_stats()
for i, query in enumerate(queries[:20]):  # Same 20 queries
    dists, indices = index.search(query, data, k=10, search_internal_result_num=64)

print("\n" + "="*80)
print("WARM CACHE RESULTS (same 20 queries)")
index.print_detailed_stats()

# Cleanup
import shutil
shutil.rmtree('./profile_index', ignore_errors=True)
