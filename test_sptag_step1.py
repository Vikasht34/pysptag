"""
Test SPTAG-style configuration on SIFT 100K
Step-by-step validation
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("="*80)
print("SPTAG-Style Test on SIFT1M")
print("="*80)

# Load data
print("\nLoading SIFT1M...")
base = np.fromfile('data/sift/sift_base.fvecs', dtype=np.float32)
base = base.reshape(-1, 129)[:, 1:].copy()  # Full 1M
print(f"  Base: {base.shape}")

queries = np.fromfile('data/sift/sift_query.fvecs', dtype=np.float32)
queries = queries.reshape(-1, 129)[:, 1:].copy()
print(f"  Queries: {queries.shape}")

groundtruth = np.fromfile('data/sift/sift_groundtruth.ivecs', dtype=np.int32)
groundtruth = groundtruth.reshape(-1, 101)[:, 1:].copy()

# Build with SPTAG parameters
print("\nBuilding index (SPTAG parameters):")
print("  ratio=0.01 (1% of data as centroids)")
print("  replica=8")
print("  posting_limit=118")
print("  no quantization")

index = SPANNDiskOptimized(
    dim=128,
    target_posting_size=118,  # SPTAG default
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    tree_type='KDT',
    disk_path='./sift1m_sptag',
    cache_size=512,
    clustering='kmeans'
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Built in {build_time:.1f}s")

# Warm up
print("\nWarming up...")
for q in queries[:20]:
    index.search(q, base, k=10, search_internal_result_num=64, max_check=4096)

# Test
print("\nTesting (100 queries, max_check=4096)...")
recalls = []
latencies = []

for i, q in enumerate(queries[:100]):
    t0 = time.time()
    dists, indices = index.search(q, base, k=10, 
                                  search_internal_result_num=64,
                                  max_check=4096)
    latencies.append((time.time() - t0) * 1000)
    
    gt = set(groundtruth[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)

p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
recall = np.mean(recalls) * 100

print(f"\n{'='*80}")
print("RESULTS")
print("="*80)
print(f"Latency: p50={p50:.2f}ms, p90={p90:.2f}ms")
print(f"Recall@10: {recall:.1f}%")
print(f"Clusters: {index.num_clusters}")
print(f"Build time: {build_time:.1f}s")

if p90 < 10 and recall >= 90:
    print(f"\n✅ SUCCESS!")
elif recall >= 90:
    print(f"\n⚠️  Good recall but slow (p90={p90:.2f}ms)")
elif p90 < 10:
    print(f"\n⚠️  Fast but low recall ({recall:.1f}%)")
else:
    print(f"\n❌ Both metrics need improvement")

# Cleanup
import shutil
shutil.rmtree('./sift1m_sptag', ignore_errors=True)
