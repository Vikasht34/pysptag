"""
Test optimized SPANN on SIFT1M dataset
Target: p90 <10ms with 90%+ recall
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("="*80)
print("SIFT 100K Test - Optimized Configuration")
print("="*80)

# Load SIFT1M (use 100K subset for testing)
print("\nLoading SIFT dataset (100K subset)...")
base = np.fromfile('data/sift/sift_base.fvecs', dtype=np.float32)
base = base.reshape(-1, 129)[:, 1:][:100000].copy()  # 100K vectors
print(f"  Base: {base.shape}")

queries = np.fromfile('data/sift/sift_query.fvecs', dtype=np.float32)
queries = queries.reshape(-1, 129)[:, 1:].copy()
print(f"  Queries: {queries.shape}")

groundtruth = np.fromfile('data/sift/sift_groundtruth.ivecs', dtype=np.int32)
groundtruth = groundtruth.reshape(-1, 101)[:, 1:].copy()
print(f"  Groundtruth: {groundtruth.shape}")

# Build index
print("\nBuilding index...")
print("  Config: target_posting=1000 (100 clusters), replica=8, no_quant")

index = SPANNDiskOptimized(
    dim=128,
    target_posting_size=1000,  # 100K / 1000 = 100 clusters
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    tree_type='KDT',
    disk_path='./sift100k_optimized',
    cache_size=512,
    clustering='kmeans'
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")

# Warm up
print("\nWarming up...")
for q in queries[:20]:
    index.search(q, base, k=10, search_internal_result_num=64, max_check=32768)

# Test
print("\nTesting (100 queries)...")
recalls = []
latencies = []

for i, q in enumerate(queries[:100]):
    t0 = time.time()
    dists, indices = index.search(q, base, k=10, 
                                  search_internal_result_num=64,
                                  max_check=32768)  # Increased for 100K
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
print(f"  Build time: {build_time:.1f}s")
print(f"  QPS: {1000/p50:.1f}")

if p90 < 10 and recall >= 90:
    print(f"\n✅ SUCCESS: p90={p90:.2f}ms with {recall:.1f}% recall")
else:
    print(f"\n⚠️  Needs tuning:")
    if p90 >= 10:
        print(f"   - p90 too high: {p90:.2f}ms")
    if recall < 90:
        print(f"   - Recall too low: {recall:.1f}%")

# Cleanup
import shutil
shutil.rmtree('./sift100k_optimized', ignore_errors=True)
