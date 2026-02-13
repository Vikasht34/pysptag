"""
Compare optimized vs original disk-based SPANN
Tests Phase 1 optimizations: batch I/O + mmap + cache
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, '.')

print("="*80)
print("Optimized vs Original Disk-Based SPANN")
print("="*80)

# Generate test data (10K vectors for quick test)
np.random.seed(42)
data = np.random.randn(10000, 768).astype(np.float32)
queries = np.random.randn(100, 768).astype(np.float32)

# Ground truth
print("\nComputing ground truth...")
gt_indices = []
for query in queries:
    dists = np.sum((data - query) ** 2, axis=1)
    gt_indices.append(np.argsort(dists)[:10])
gt_indices = np.array(gt_indices)

results = []

# Test 1: Original
print("\n" + "="*80)
print("TEST 1: Original Disk-Based")
print("="*80)

from src.index.spann_disk import SPANNDiskBased

t0 = time.time()
index1 = SPANNDiskBased(
    dim=768,
    target_posting_size=500,
    replica_count=6,
    bq=4,
    use_rabitq=True,
    tree_type='KDT',
    disk_path='./test_original'
)
index1.build(data)
build_time1 = time.time() - t0
print(f"Build time: {build_time1:.2f}s")

t0 = time.time()
recalls1 = []
latencies1 = []
for i, query in enumerate(queries):
    q_start = time.time()
    dists, indices = index1.search(query, data, k=10)
    latencies1.append((time.time() - q_start) * 1000)
    
    gt = set(gt_indices[i])
    found = set(indices[:10])
    recalls1.append(len(gt & found) / 10)

search_time1 = time.time() - t0
avg_recall1 = np.mean(recalls1) * 100
p50_1 = np.percentile(latencies1, 50)
p90_1 = np.percentile(latencies1, 90)
qps1 = len(queries) / search_time1

print(f"Search: {search_time1:.2f}s, QPS: {qps1:.1f}")
print(f"Latency: p50={p50_1:.2f}ms, p90={p90_1:.2f}ms")
print(f"Recall@10: {avg_recall1:.2f}%")

# Test 2: Optimized
print("\n" + "="*80)
print("TEST 2: Optimized (Batch I/O + mmap + cache)")
print("="*80)

from src.index.spann_disk_optimized import SPANNDiskOptimized

t0 = time.time()
index2 = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=6,
    bq=4,
    use_rabitq=True,
    tree_type='KDT',
    disk_path='./test_optimized',
    cache_size=128
)
index2.build(data)
build_time2 = time.time() - t0
print(f"Build time: {build_time2:.2f}s")

t0 = time.time()
recalls2 = []
latencies2 = []
for i, query in enumerate(queries):
    q_start = time.time()
    dists, indices = index2.search(query, data, k=10)
    latencies2.append((time.time() - q_start) * 1000)
    
    gt = set(gt_indices[i])
    found = set(indices[:10])
    recalls2.append(len(gt & found) / 10)

search_time2 = time.time() - t0
avg_recall2 = np.mean(recalls2) * 100
p50_2 = np.percentile(latencies2, 50)
p90_2 = np.percentile(latencies2, 90)
qps2 = len(queries) / search_time2

print(f"Search: {search_time2:.2f}s, QPS: {qps2:.1f}")
print(f"Latency: p50={p50_2:.2f}ms, p90={p90_2:.2f}ms")
print(f"Recall@10: {avg_recall2:.2f}%")
index2.print_cache_stats()

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Version':<20} {'p50(ms)':<10} {'p90(ms)':<10} {'QPS':<10} {'Speedup':<10}")
print("-"*80)
print(f"{'Original':<20} {p50_1:<10.2f} {p90_1:<10.2f} {qps1:<10.1f} {'1.0×':<10}")
print(f"{'Optimized':<20} {p50_2:<10.2f} {p90_2:<10.2f} {qps2:<10.1f} {f'{qps2/qps1:.1f}×':<10}")
print("="*80)
print(f"\nSpeedup: {qps2/qps1:.1f}× faster")
print(f"Latency reduction: {p50_1:.2f}ms → {p50_2:.2f}ms ({(1-p50_2/p50_1)*100:.1f}% faster)")

# Cleanup
import shutil
shutil.rmtree('./test_original', ignore_errors=True)
shutil.rmtree('./test_optimized', ignore_errors=True)
