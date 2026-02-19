"""
Test HNSW centroid search vs BKT+RNG on SIFT dataset
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_DIR = '/Users/viktari/pysptag/data/sift'
NUM_QUERIES = 100

print("="*80)
print("HNSW vs BKT+RNG Centroid Search Comparison - SIFT 1M")
print("="*80)

# Load data
print("\nLoading SIFT data...")
base = np.fromfile(f'{DATA_DIR}/sift_base.fvecs', dtype=np.float32)
base = base.reshape(-1, 129)[:, 1:].copy()  # Skip first value (dimension)
queries = np.fromfile(f'{DATA_DIR}/sift_query.fvecs', dtype=np.float32)
queries = queries.reshape(-1, 129)[:, 1:].copy()[:NUM_QUERIES]
gt_raw = np.fromfile(f'{DATA_DIR}/sift_groundtruth.ivecs', dtype=np.int32)
groundtruth = gt_raw.reshape(-1, 101)[:, 1:].copy()[:NUM_QUERIES]  # Skip first value

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Test 1: BKT+RNG (original)
print("\n" + "="*80)
print("Test 1: BKT+RNG Centroid Search")
print("="*80)

index_bkt = SPANNDiskOptimized(
    dim=128,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    tree_type='BKT',
    use_faiss_centroids=False,
    use_hnsw_centroids=False,
    disk_path='/tmp/sift_bkt_test',
    cache_size=2000
)

print("Building index with BKT+RNG...")
t0 = time.time()
index_bkt.build(base)
build_time_bkt = time.time() - t0
print(f"✓ Built in {build_time_bkt:.1f}s")

# Test 2: HNSW
print("\n" + "="*80)
print("Test 2: HNSW Centroid Search")
print("="*80)

index_hnsw = SPANNDiskOptimized(
    dim=128,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    use_faiss_centroids=False,
    use_hnsw_centroids=True,
    hnsw_m=16,
    hnsw_ef_construction=200,
    disk_path='/tmp/sift_hnsw_test',
    cache_size=2000
)

print("Building index with HNSW...")
t0 = time.time()
index_hnsw.build(base)
build_time_hnsw = time.time() - t0
print(f"✓ Built in {build_time_hnsw:.1f}s")

# Compare search performance
print("\n" + "="*80)
print("Search Performance Comparison")
print("="*80)

for name, index in [('BKT+RNG', index_bkt), ('HNSW', index_hnsw)]:
    print(f"\n{name}:")
    
    # Warmup
    for q in queries[:10]:
        index.search(q, base, k=100, search_internal_result_num=64, max_check=10000)
    
    # Test
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        indices, dists = index.search(
            query, base, k=100,
            search_internal_result_num=64,
            max_check=10000
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        
        gt = set(groundtruth[i][:100])
        found = set(indices[:100])
        recalls.append(len(gt & found) / 100)
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    avg = np.mean(latencies)
    
    print(f"  Recall@100: {avg_recall:.2f}%")
    print(f"  P50 latency: {p50:.2f}ms")
    print(f"  P90 latency: {p90:.2f}ms")
    print(f"  Avg latency: {avg:.2f}ms")

print("\n" + "="*80)
print("Build Time Comparison:")
print(f"  BKT+RNG: {build_time_bkt:.1f}s")
print(f"  HNSW: {build_time_hnsw:.1f}s")
print(f"  Speedup: {build_time_bkt/build_time_hnsw:.2f}x")
print("="*80)
