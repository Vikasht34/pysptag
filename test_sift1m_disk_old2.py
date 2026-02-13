"""
Test SPTAG-optimized disk-based SPANN on SIFT1M
Uses ratio-based clustering (1% of data as centroids)
Target: p90 <10ms with 90%+ recall on EBS
"""
import numpy as np
import time
import os

print("="*80)
print("SIFT1M Disk-Based Test (SPTAG-Optimized)")
print("="*80)

# Load SIFT1M
print("\nLoading SIFT1M...")
base = np.fromfile('data/sift/sift_base.fvecs', dtype=np.float32)
base = base.reshape(-1, 129)[:, 1:].copy()
print(f"  Base: {base.shape}")

queries = np.fromfile('data/sift/sift_query.fvecs', dtype=np.float32)
queries = queries.reshape(-1, 129)[:, 1:].copy()
print(f"  Queries: {queries.shape}")

groundtruth = np.fromfile('data/sift/sift_groundtruth.ivecs', dtype=np.int32)
groundtruth = groundtruth.reshape(-1, 101)[:, 1:].copy()

# Build index
print("\nBuilding SPTAG-optimized index...")
print("  Config: ratio=0.01 (10K clusters), replica=8, no_quant")
print("  Posting limit: 500 (adaptive for 1M)")

from src.index.spann_disk_optimized import SPANNDiskOptimized

index = SPANNDiskOptimized(
    dim=128,
    target_posting_size=500,  # Adaptive limit for 1M vectors
    replica_count=8,
    use_rabitq=False,  # No quantization
    metric='L2',
    tree_type='KDT',
    disk_path='./sift1m_disk_sptag',
    cache_size=1024,
    clustering='kmeans'
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")

# Warm up cache
print("\nWarming up cache (20 queries)...")
for q in queries[:20]:
    index.search(q, base, k=10, search_internal_result_num=64, max_check=8192)

# Test
print("\nTesting (100 queries)...")
recalls = []
latencies = []

for i, q in enumerate(queries[:100]):
    t0 = time.time()
    dists, indices = index.search(q, base, k=10, 
                                  search_internal_result_num=64,
                                  max_check=8192)
    latencies.append((time.time() - t0) * 1000)
    
    gt = set(groundtruth[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/100 queries...")

p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p99 = np.percentile(latencies, 99)
recall = np.mean(recalls) * 100

print(f"\n{'='*80}")
print("RESULTS")
print("="*80)
print(f"Latency:")
print(f"  p50: {p50:.2f}ms")
print(f"  p90: {p90:.2f}ms {'✅' if p90 < 10 else '❌'} (target <10ms)")
print(f"  p99: {p99:.2f}ms")
print(f"\nRecall@10: {recall:.1f}% {'✅' if recall >= 90 else '❌'} (target ≥90%)")
print(f"\nIndex stats:")
print(f"  Clusters: {index.num_clusters}")
print(f"  Build time: {build_time:.1f}s ({build_time/60:.1f} min)")
print(f"  QPS: {1000/p50:.1f}")
print(f"  Cache hit rate: {index._cache_hits/(index._cache_hits+index._cache_misses)*100:.1f}%")

# Check disk usage
disk_size = sum(os.path.getsize(os.path.join(dp, f)) 
                for dp, dn, fn in os.walk('./sift1m_disk_sptag') 
                for f in fn) / (1024**3)
print(f"  Disk usage: {disk_size:.2f} GB")

if p90 < 10 and recall >= 90:
    print(f"\n✅ SUCCESS: p90={p90:.2f}ms with {recall:.1f}% recall")
else:
    print(f"\n⚠️  Needs tuning:")
    if p90 >= 10:
        print(f"   - p90 too high: {p90:.2f}ms (target <10ms)")
    if recall < 90:
        print(f"   - Recall too low: {recall:.1f}% (target ≥90%)")
        print(f"   - Try: increase max_check or reduce posting_limit")

print(f"\n{'='*80}")
print("To cleanup: rm -rf ./sift1m_disk_sptag")
print("="*80)
