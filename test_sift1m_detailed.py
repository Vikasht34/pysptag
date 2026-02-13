#!/usr/bin/env python3
"""
SIFT 1M test with detailed latency breakdown at every level
"""
import numpy as np
import time
from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

def load_ivecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            d = np.fromfile(f, dtype=np.int32, count=1)
            if len(d) == 0:
                break
            vec = np.fromfile(f, dtype=np.int32, count=d[0])
            data.append(vec)
    return np.array(data)

# Load data
print("Loading SIFT 1M dataset...")
base = load_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs')
queries = load_fvecs('/Users/viktari/pysptag/data/sift/sift_query.fvecs')
groundtruth = load_ivecs('/Users/viktari/pysptag/data/sift/sift_groundtruth.ivecs')

# Load index
print("\nLoading index...")
import pickle
index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    tree_type='BKT',
    replica_count=8,
    use_rng_filtering=True,
    clustering='hierarchical',
    target_posting_size=800,
    disk_path='/tmp/sift1m_test',
    use_rabitq=False,
    preload_postings=False
)

with open('/tmp/sift1m_test/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    index.centroids = metadata['centroids']
    index.tree = metadata['tree']
    index.rng = metadata['rng']
    index.num_clusters = metadata['num_clusters']

print(f"Loaded: {index.num_clusters} clusters")
print(f"RNG graph size: {len(index.rng.graph)}")
print(f"RNG graph[0]: {index.rng.graph[0][:5]}...")

# Detailed search breakdown
print("\n" + "="*70)
print("DETAILED LATENCY BREAKDOWN")
print("="*70)

num_queries = 100
k = 10
max_check = 64  # SPTAG default, not 4096!

# Timing arrays
centroid_search_times = []
posting_load_times = []
distance_comp_times = []
sort_times = []
total_times = []
bytes_read_list = []
recalls = []

for i in range(num_queries):
    query = queries[i]
    index._bytes_read = 0
    
    t_total = time.perf_counter()
    
    # === STAGE 1: Centroid Search ===
    t0 = time.perf_counter()
    if hasattr(index, 'rng') and len(index.rng.graph) > 0 and index.rng.graph[0][0] >= 0:
        # Use BKTree+RNG search (SPTAG-style with early termination)
        from src.core.bktree_rng_search import bktree_rng_search
        nearest_centroids = bktree_rng_search(
            query, index.centroids, index.tree.tree_roots,
            index.tree.tree_start, index.rng.graph,
            64, index.metric, initial_candidates=100, max_check=500
        )
    else:
        # Fallback to faiss
        import faiss
        if not hasattr(index, '_centroid_index'):
            index._centroid_index = faiss.IndexFlatL2(index.centroids.shape[1])
            index._centroid_index.add(index.centroids.astype(np.float32))
        _, nearest_centroids = index._centroid_index.search(query.reshape(1, -1).astype(np.float32), 64)
        nearest_centroids = nearest_centroids[0]
    t_centroid = (time.perf_counter() - t0) * 1000
    
    # === STAGE 2: Load Posting Lists ===
    t0 = time.perf_counter()
    candidates = []
    for cid in nearest_centroids[:min(max_check, len(nearest_centroids))]:
        posting = index._load_posting_mmap(cid)
        if posting is not None and posting[0] is not None:
            candidates.extend(posting[0])  # Load ALL vectors
    t_posting = (time.perf_counter() - t0) * 1000
    
    # === STAGE 3: Distance Computation ===
    t0 = time.perf_counter()
    candidates = list(set(candidates))
    if len(candidates) > 0:
        candidate_vecs = base[candidates]
        dists = np.sum((candidate_vecs - query) ** 2, axis=1)
    else:
        dists = np.array([])
    t_distance = (time.perf_counter() - t0) * 1000
    
    # === STAGE 4: Sort and Select Top-K ===
    t0 = time.perf_counter()
    if len(dists) >= k:
        top_k_idx = np.argpartition(dists, k)[:k]
        top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
        result_ids = [candidates[idx] for idx in top_k_idx]
    else:
        result_ids = candidates
    t_sort = (time.perf_counter() - t0) * 1000
    
    t_total_elapsed = (time.perf_counter() - t_total) * 1000
    
    # Metrics
    centroid_search_times.append(t_centroid)
    posting_load_times.append(t_posting)
    distance_comp_times.append(t_distance)
    sort_times.append(t_sort)
    total_times.append(t_total_elapsed)
    bytes_read_list.append(index._bytes_read)
    
    recall = len(set(result_ids) & set(int(x) for x in groundtruth[i][:k])) / k
    recalls.append(recall)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{num_queries} queries done")

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Queries:           {num_queries}")
print(f"Recall@10:         {np.mean(recalls)*100:.2f}%")
print(f"\nTotal Latency:")
print(f"  p50:             {np.percentile(total_times, 50):.2f} ms")
print(f"  p90:             {np.percentile(total_times, 90):.2f} ms")
print(f"  p99:             {np.percentile(total_times, 99):.2f} ms")
print(f"  mean:            {np.mean(total_times):.2f} ms")

print(f"\n{'Stage':<25} {'Mean (ms)':<12} {'p50 (ms)':<12} {'p99 (ms)':<12} {'% of Total':<12}")
print("-"*70)

total_mean = np.mean(total_times)
stages = [
    ('1. Centroid Search', centroid_search_times),
    ('2. Posting Load', posting_load_times),
    ('3. Distance Computation', distance_comp_times),
    ('4. Sort & Select', sort_times)
]

for name, times in stages:
    mean_t = np.mean(times)
    p50_t = np.percentile(times, 50)
    p99_t = np.percentile(times, 99)
    pct = (mean_t / total_mean) * 100
    print(f"{name:<25} {mean_t:>10.2f}   {p50_t:>10.2f}   {p99_t:>10.2f}   {pct:>10.1f}%")

print(f"\nDisk I/O:")
print(f"  Avg per query:   {np.mean(bytes_read_list)/1024:.1f} KB")
print(f"  Total:           {np.sum(bytes_read_list)/1024/1024:.1f} MB")
print("="*70)
