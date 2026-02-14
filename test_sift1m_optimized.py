#!/usr/bin/env python3
"""
Optimized SIFT 1M test with all optimizations:
- Faiss centroid search (~0.5ms)
- RaBitQNumba 2-bit quantization (16× compression)
- Small posting size (100 vectors)
- Reduced centroids searched (32)
"""
import numpy as np
import time
from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Load SIFT 1M
print("Loading SIFT 1M dataset...")
base = load_fvecs('data/sift/sift_base.fvecs')
queries = load_fvecs('data/sift/sift_query.fvecs')
ground_truth = load_fvecs('data/sift/sift_groundtruth.ivecs').astype(np.int32)

print(f"Base: {base.shape}")
print(f"Queries: {queries.shape}")
print(f"Ground truth: {ground_truth.shape}")

# Build optimized index
print("\n" + "="*70)
print("BUILDING OPTIMIZED INDEX")
print("="*70)

index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    
    # Optimizations
    use_faiss_centroids=True,      # Fast centroid search
    tree_type='FLAT',               # Not needed with faiss
    use_rabitq=True,                # 2-bit quantization
    target_posting_size=100,        # Small postings for low latency
    replica_count=8,                # SPTAG default
    
    # Clustering
    clustering='hierarchical',      # SPTAG-style
    use_rng_filtering=True,         # NPA filtering
    
    # Storage
    disk_path='/tmp/sift1m_optimized',
    preload_postings=False,
    num_threads=1
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Build time: {build_time:.1f}s")

# Test search
print("\n" + "="*70)
print("TESTING SEARCH (100 queries)")
print("="*70)

recalls = []
latencies = []

for i, query in enumerate(queries[:100]):
    t0 = time.perf_counter()
    ids, dists = index.search(
        query, base, 
        k=10,
        search_internal_result_num=32,  # Reduced from 64
        max_check=4096
    )
    latency = (time.perf_counter() - t0) * 1000
    latencies.append(latency)
    
    # Calculate recall
    gt = set(ground_truth[i, :10])
    pred = set(ids[:10])
    recall = len(gt & pred) / 10.0
    recalls.append(recall)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/100 queries done")

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Queries:           100")
print(f"Recall@10:         {np.mean(recalls)*100:.2f}%")
print(f"\nLatency:")
print(f"  p50:             {np.percentile(latencies, 50):.2f} ms")
print(f"  p90:             {np.percentile(latencies, 90):.2f} ms")
print(f"  p99:             {np.percentile(latencies, 99):.2f} ms")
print(f"  mean:            {np.mean(latencies):.2f} ms")
print(f"\nDisk I/O:")
print(f"  Avg per query:   {index._bytes_read / 100 / 1024:.1f} KB")
print(f"  Total:           {index._bytes_read / 1024**2:.1f} MB")
print("="*70)

# Target check
mean_latency = np.mean(latencies)
mean_recall = np.mean(recalls) * 100

print(f"\n🎯 TARGET CHECK:")
print(f"  Latency: {mean_latency:.1f}ms {'✅' if mean_latency < 10 else '❌'} (target: <10ms)")
print(f"  Recall:  {mean_recall:.1f}% {'✅' if mean_recall > 90 else '❌'} (target: >90%)")

if mean_latency < 10 and mean_recall > 90:
    print("\n🎉 SUCCESS! Both targets achieved!")
else:
    print("\n⚠️  Needs tuning:")
    if mean_latency >= 10:
        print("  - Reduce search_internal_result_num or target_posting_size")
    if mean_recall <= 90:
        print("  - Increase search_internal_result_num or target_posting_size")
