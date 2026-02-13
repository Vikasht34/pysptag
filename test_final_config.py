"""
Final optimized configuration based on analysis.

Key findings:
1. No quantization is faster (posting search bottleneck)
2. max_check=8192 needed for 90%+ recall
3. posting_size=118 with limits reduces disk I/O
4. Hierarchical clustering is too slow - use k-means with limits

Target: p90 <10ms with 90%+ recall on EBS
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("="*80)
print("FINAL OPTIMIZED CONFIGURATION")
print("="*80)

# Test data
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

# Build with optimal config
print("\nBuilding index with optimal parameters:")
print("  posting_size=200 (balanced)")
print("  replica=8")
print("  no quantization")
print("  search_centroids=64")
print("  max_check=8192")

index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=200,  # Larger postings for better recall
    replica_count=8,
    bq=4,
    use_rabitq=False,  # No quantization
    metric='L2',
    tree_type='KDT',
    disk_path='./final_optimized',
    cache_size=256,
    clustering='kmeans'  # Fast k-means with posting limits
)

t0 = time.time()
index.build(data)
build_time = time.time() - t0
print(f"\n✓ Built in {build_time:.2f}s")

# Warm up
print("\nWarming up cache...")
for query in queries[:10]:
    index.search(query, data, k=10, search_internal_result_num=64, max_check=8192)

# Test
print("\nTesting (100 queries)...")
recalls = []
latencies = []

for i, query in enumerate(queries):
    t0 = time.time()
    dists, indices = index.search(query, data, k=10, 
                                  search_internal_result_num=64,
                                  max_check=8192)
    latencies.append((time.time() - t0) * 1000)
    
    gt = set(gt_indices[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)

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
print(f"\nClusters: {index.num_clusters}")
print(f"Build time: {build_time:.2f}s")
print(f"QPS: {1000/p50:.1f}")

if p90 < 10 and recall >= 90:
    print(f"\n{'='*80}")
    print("✅ SUCCESS: Target achieved!")
    print(f"   p90={p90:.2f}ms with {recall:.1f}% recall")
    print("="*80)
    print("\nNext steps:")
    print("1. Test on SIFT1M dataset")
    print("2. Deploy to EC2 with EBS")
    print("3. Test on Cohere 1M embeddings")
else:
    print(f"\n{'='*80}")
    print("⚠️  Needs further optimization")
    if p90 >= 10:
        print(f"   p90 too high: {p90:.2f}ms (need <10ms)")
    if recall < 90:
        print(f"   Recall too low: {recall:.1f}% (need ≥90%)")
    print("="*80)

# Cleanup
import shutil
shutil.rmtree('./final_optimized', ignore_errors=True)
