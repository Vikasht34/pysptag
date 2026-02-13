"""
Test hierarchical clustering vs k-means for p90 <10ms target
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("="*80)
print("Hierarchical vs K-Means Clustering Test")
print("Target: p90 <10ms with 90%+ recall")
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

def test_config(name, clustering, posting_size, replica, search_centroids, max_check):
    """Test a configuration"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"  clustering={clustering}, posting={posting_size}, replica={replica}")
    print(f"  search_centroids={search_centroids}, max_check={max_check}")
    print("="*80)
    
    # Build
    index = SPANNDiskOptimized(
        dim=768,
        target_posting_size=posting_size,
        replica_count=replica,
        bq=4,
        use_rabitq=False,
        metric='L2',
        tree_type='KDT',
        disk_path=f'./test_{name.replace(" ", "_")}',
        cache_size=256,
        clustering=clustering
    )
    
    t0 = time.time()
    index.build(data)
    build_time = time.time() - t0
    
    # Warm up
    for query in queries[:10]:
        index.search(query, data, k=10, 
                    search_internal_result_num=search_centroids,
                    max_check=max_check)
    
    # Test
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries):
        t0 = time.time()
        dists, indices = index.search(query, data, k=10, 
                                      search_internal_result_num=search_centroids,
                                      max_check=max_check)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    recall = np.mean(recalls) * 100
    
    print(f"\nResults:")
    print(f"  Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
    print(f"  Recall: {recall:.1f}%")
    print(f"  Clusters: {index.num_clusters}")
    print(f"  Build time: {build_time:.2f}s")
    
    # Cleanup
    import shutil
    shutil.rmtree(f'./test_{name.replace(" ", "_")}', ignore_errors=True)
    
    return {
        'name': name,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': recall,
        'clusters': index.num_clusters,
        'build_time': build_time
    }

# Test configurations
results = []

# 1. K-means baseline
results.append(test_config(
    "K-Means Baseline",
    clustering='kmeans',
    posting_size=118,
    replica=8,
    search_centroids=64,
    max_check=8192
))

# 2. Hierarchical clustering
results.append(test_config(
    "Hierarchical (SPTAG)",
    clustering='hierarchical',
    posting_size=118,
    replica=8,
    search_centroids=64,
    max_check=8192
))

# 3. Hierarchical with fewer centroids
results.append(test_config(
    "Hierarchical + Fewer Centroids",
    clustering='hierarchical',
    posting_size=118,
    replica=8,
    search_centroids=32,
    max_check=8192
))

# 4. Hierarchical with fewer replicas
results.append(test_config(
    "Hierarchical + Fewer Replicas",
    clustering='hierarchical',
    posting_size=118,
    replica=6,
    search_centroids=64,
    max_check=8192
))

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"{'Config':<35} {'p90(ms)':<10} {'Recall%':<10} {'Clusters':<10} {'Status':<15}")
print("-"*80)

for r in results:
    status = ""
    if r['p90'] < 10 and r['recall'] >= 90:
        status = "✅ TARGET MET"
    elif r['p90'] < 10:
        status = "⚠️  Fast, low recall"
    elif r['recall'] >= 90:
        status = "⚠️  Good recall, slow"
    else:
        status = "❌ Needs work"
    
    print(f"{r['name']:<35} {r['p90']:<10.2f} {r['recall']:<10.1f} {r['clusters']:<10} {status:<15}")

print("="*80)

# Find best
best = min([r for r in results if r['recall'] >= 90], 
           key=lambda x: x['p90'], default=None)

if best:
    print(f"\n✅ BEST: {best['name']}")
    print(f"   p90={best['p90']:.2f}ms, recall={best['recall']:.1f}%")
else:
    print(f"\n❌ No config met target")
