#!/usr/bin/env python3
"""
Test SPTAG-exact implementation on SIFT 1M
Matches SPTAG parameters exactly for apple-to-apple comparison
"""
import numpy as np
import time
from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized
from src.core.sptag_params import (
    DEFAULT_REPLICA_COUNT,
    DEFAULT_INTERNAL_RESULT_NUM,
    DEFAULT_MAX_CHECK,
    DEFAULT_MAX_DIST_RATIO,
    DEFAULT_POSTING_VECTOR_LIMIT,
    DEFAULT_POSTING_PAGE_LIMIT
)

print("="*70)
print("SPTAG-EXACT IMPLEMENTATION TEST")
print("="*70)

# Load SIFT 1M
print("\n[1/4] Loading SIFT 1M dataset...")
base = load_fvecs('data/sift/sift_base.fvecs')
queries = load_fvecs('data/sift/sift_query.fvecs')
ground_truth = load_fvecs('data/sift/sift_groundtruth.ivecs').astype(np.int32)

print(f"  Base: {base.shape}")
print(f"  Queries: {queries.shape}")
print(f"  Ground truth: {ground_truth.shape}")

# Build index with SPTAG-exact parameters
print("\n[2/4] Building index with SPTAG-exact parameters...")
print(f"  replica_count = {DEFAULT_REPLICA_COUNT}")
print(f"  posting_vector_limit = {DEFAULT_POSTING_VECTOR_LIMIT}")
print(f"  posting_page_limit = {DEFAULT_POSTING_PAGE_LIMIT}")
print(f"  internal_result_num = {DEFAULT_INTERNAL_RESULT_NUM}")

index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    
    # SPTAG-exact parameters
    replica_count=DEFAULT_REPLICA_COUNT,
    posting_vector_limit=DEFAULT_POSTING_VECTOR_LIMIT,
    posting_page_limit=DEFAULT_POSTING_PAGE_LIMIT,
    internal_result_num=DEFAULT_INTERNAL_RESULT_NUM,
    
    # Optimizations
    use_rabitq=False,  # Start without quantization
    use_faiss_centroids=False,  # Start with BKTree+RNG
    tree_type='BKT',
    clustering='hierarchical',
    use_rng_filtering=True,
    
    disk_path='/tmp/sift1m_sptag_exact',
    preload_postings=False
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Build time: {build_time:.1f}s")

# Test search with SPTAG-exact parameters
print("\n[3/4] Testing search (100 queries)...")
print(f"  search_internal_result_num = {DEFAULT_INTERNAL_RESULT_NUM}")
print(f"  max_check = {DEFAULT_MAX_CHECK}")
print(f"  max_dist_ratio = {DEFAULT_MAX_DIST_RATIO}")

recalls = []
latencies = []
centroid_times = []
posting_times = []
distance_times = []

for i, query in enumerate(queries[:100]):
    # Reset bytes counter
    index._bytes_read = 0
    
    t0 = time.perf_counter()
    ids, dists = index.search(
        query, base,
        k=10,
        search_internal_result_num=DEFAULT_INTERNAL_RESULT_NUM,
        max_check=DEFAULT_MAX_CHECK,
        max_dist_ratio=DEFAULT_MAX_DIST_RATIO
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
print("\n[4/4] Results")
print("="*70)
print(f"Configuration:")
print(f"  Posting size: {index.target_posting_size} vectors")
print(f"  Quantization: {'RaBitQ' if index.use_rabitq else 'None'}")
print(f"  Centroid search: {'Faiss' if index.use_faiss_centroids else 'BKTree+RNG'}")
print()
print(f"Performance:")
print(f"  Recall@10:   {np.mean(recalls)*100:.2f}%")
print(f"  Latency p50: {np.percentile(latencies, 50):.2f} ms")
print(f"  Latency p90: {np.percentile(latencies, 90):.2f} ms")
print(f"  Latency p99: {np.percentile(latencies, 99):.2f} ms")
print(f"  Latency avg: {np.mean(latencies):.2f} ms")
print()
print(f"Disk I/O:")
print(f"  Avg per query: {index._bytes_read / 100 / 1024:.1f} KB")
print("="*70)

# Compare to SPTAG expected performance
print("\nSPTAG Expected Performance (from paper):")
print("  Recall@10:   ~95%")
print("  Latency:     ~5-10ms (with SSD)")
print()
print("Our Results:")
if np.mean(recalls) * 100 >= 90:
    print(f"  Recall:  ✅ {np.mean(recalls)*100:.1f}% (target: >90%)")
else:
    print(f"  Recall:  ❌ {np.mean(recalls)*100:.1f}% (target: >90%)")

if np.mean(latencies) <= 50:
    print(f"  Latency: ✅ {np.mean(latencies):.1f}ms (reasonable for Python)")
else:
    print(f"  Latency: ⚠️  {np.mean(latencies):.1f}ms (can optimize)")
