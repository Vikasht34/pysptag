#!/usr/bin/env python3
"""
Optimized SPTAG implementation with all improvements:
1. Larger posting size (200 vectors) for better recall
2. Faiss centroid search (0.5ms vs 8ms)
3. RaBitQ 2-bit quantization (16× compression)
"""
import numpy as np
import time
import struct

def load_ivecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            data.append(vec)
    return np.array(data, dtype=np.int32)

from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("="*70)
print("OPTIMIZED SPTAG IMPLEMENTATION")
print("="*70)

# Load SIFT 1M
print("\n[1/4] Loading SIFT 1M...")
base = load_fvecs('data/sift/sift_base.fvecs')
queries = load_fvecs('data/sift/sift_query.fvecs')
ground_truth = load_ivecs('data/sift/sift_groundtruth.ivecs')
print(f"  Loaded {len(base)} vectors")

# Build optimized index
print("\n[2/4] Building optimized index...")
print("  Optimizations:")
print("    - Posting size: 200 vectors (vs 119 SPTAG default)")
print("    - Faiss centroids: Fast search")
print("    - RaBitQ: 2-bit quantization")

index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    
    # Optimized parameters
    target_posting_size=200,        # Larger for better recall
    replica_count=8,
    
    # Optimizations
    use_faiss_centroids=True,       # Fast centroid search
    use_rabitq=True,                # 2-bit quantization
    bq=4,
    
    # SPTAG settings
    tree_type='BKT',
    clustering='hierarchical',
    use_rng_filtering=True,
    
    disk_path='/tmp/sift1m_optimized',
    preload_postings=False
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Build time: {build_time:.1f}s")

# Test search
print("\n[3/4] Testing search (100 queries)...")
recalls = []
latencies = []

for i, query in enumerate(queries[:100]):
    t0 = time.perf_counter()
    ids, dists = index.search(
        query, base, 
        k=10,
        search_internal_result_num=32,  # Reduced for speed
        max_check=4096
    )
    latency = (time.perf_counter() - t0) * 1000
    latencies.append(latency)
    
    gt = set(ground_truth[i, :10])
    pred = set(ids[:10])
    recall = len(gt & pred) / 10.0
    recalls.append(recall)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/100 done")

# Results
print("\n[4/4] Results")
print("="*70)
print(f"Recall@10:   {np.mean(recalls)*100:.2f}%")
print(f"Latency p50: {np.percentile(latencies, 50):.2f} ms")
print(f"Latency p90: {np.percentile(latencies, 90):.2f} ms")
print(f"Latency avg: {np.mean(latencies):.2f} ms")
print()
print(f"Index size:  {index.num_clusters} clusters")
print(f"Posting size: {index.target_posting_size} vectors")
print("="*70)

# Target check
mean_latency = np.mean(latencies)
mean_recall = np.mean(recalls) * 100

print(f"\n🎯 TARGET CHECK:")
print(f"  Latency: {mean_latency:.1f}ms {'✅' if mean_latency < 10 else '⚠️' if mean_latency < 20 else '❌'} (target: <10ms)")
print(f"  Recall:  {mean_recall:.1f}% {'✅' if mean_recall > 90 else '⚠️' if mean_recall > 80 else '❌'} (target: >90%)")

if mean_latency < 20 and mean_recall > 85:
    print("\n🎉 Good performance! Close to targets.")
elif mean_recall > 90:
    print(f"\n✅ Excellent recall! Latency can be optimized further.")
elif mean_latency < 10:
    print(f"\n✅ Excellent latency! Recall can be improved by increasing posting_size.")
