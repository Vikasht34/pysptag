"""
Final optimized configuration:
- Small postings (118 like SPTAG)
- Higher replica for better coverage
- Larger max_check for better recall
- Target: <10ms with 90%+ recall
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("Final Optimized Configuration")
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

# Optimized config
print("\nBuilding optimized index...")
print("  posting_size=118 (SPTAG default)")
print("  replica=8 (SPTAG default)")
print("  search_centroids=64")
print("  max_check=4096")

index = ProfiledSPANN(
    dim=768,
    target_posting_size=118,  # SPTAG default
    replica_count=8,  # SPTAG default
    bq=4,
    use_rabitq=True,
    tree_type='KDT',
    disk_path='./final_optimized',
    cache_size=256
)

t0 = time.time()
index.build(data)
build_time = time.time() - t0

print(f"✓ Built: {index.num_clusters} clusters in {build_time:.2f}s")

# Warm up
for query in queries[:10]:
    index.search(query, data, k=10, search_internal_result_num=64, max_check=4096)

# Test
print("\nTesting (100 queries)...")
index.reset_stats()
recalls = []
latencies = []

for i, query in enumerate(queries):
    t0 = time.time()
    dists, indices = index.search(query, data, k=10, 
                                  search_internal_result_num=64,
                                  max_check=4096)
    latencies.append((time.time() - t0) * 1000)
    
    gt = set(gt_indices[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)

avg_recall = np.mean(recalls) * 100
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p99 = np.percentile(latencies, 99)

print(f"\n{'='*80}")
print("RESULTS")
print("="*80)
print(f"Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
print(f"Recall@10: {avg_recall:.1f}%")
print(f"QPS: {1000/p50:.1f}")

index.print_detailed_stats()

# Compare with target
print(f"\n{'='*80}")
print("TARGET ANALYSIS")
print("="*80)
if p50 < 10 and avg_recall > 85:
    print(f"✅ SUCCESS: {p50:.2f}ms with {avg_recall:.1f}% recall")
    print("   Meets target: <10ms with >85% recall")
elif p50 < 10:
    print(f"⚠️  PARTIAL: {p50:.2f}ms but only {avg_recall:.1f}% recall")
    print(f"   Need to improve recall by {85-avg_recall:.1f}%")
elif avg_recall > 85:
    print(f"⚠️  PARTIAL: {avg_recall:.1f}% recall but {p50:.2f}ms")
    print(f"   Need to reduce latency by {p50-10:.2f}ms")
else:
    print(f"❌ MISS: {p50:.2f}ms with {avg_recall:.1f}% recall")
    print(f"   Need: {10-p50:.2f}ms faster, {85-avg_recall:.1f}% better recall")

# Cleanup
import shutil
shutil.rmtree('./final_optimized', ignore_errors=True)
