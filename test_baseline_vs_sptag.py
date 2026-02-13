"""
Quick comparison: Baseline vs SPTAG-style on 10K vectors
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("="*80)
print("Baseline vs SPTAG-Style Comparison (10K vectors)")
print("="*80)

# Generate test data
np.random.seed(42)
data = np.random.randn(10000, 128).astype(np.float32)
queries = np.random.randn(100, 128).astype(np.float32)

# Ground truth
print("\nComputing ground truth...")
gt_indices = []
for query in queries:
    dists = np.sum((data - query) ** 2, axis=1)
    gt_indices.append(np.argsort(dists)[:10])
gt_indices = np.array(gt_indices)

def test_config(name, target_posting_size, use_ratio):
    """Test a configuration"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print("="*80)
    
    index = SPANNDiskOptimized(
        dim=128,
        target_posting_size=target_posting_size,
        replica_count=8,
        use_rabitq=False,
        metric='L2',
        tree_type='KDT',
        disk_path=f'./test_{name.replace(" ", "_")}',
        cache_size=256,
        clustering='kmeans'
    )
    
    # Temporarily modify cluster method for baseline
    if not use_ratio:
        # Force old behavior
        old_cluster = index.clusterer.cluster
        def fixed_cluster(data, target_clusters):
            n = len(data)
            k = max(1, n // target_posting_size)
            return old_cluster(data, k)
        index.clusterer.cluster = fixed_cluster
    
    t0 = time.time()
    index.build(data)
    build_time = time.time() - t0
    
    # Warm up
    for q in queries[:10]:
        index.search(q, data, k=10, search_internal_result_num=64, max_check=4096)
    
    # Test
    recalls = []
    latencies = []
    
    for i, q in enumerate(queries):
        t0 = time.time()
        dists, indices = index.search(q, data, k=10, 
                                      search_internal_result_num=64,
                                      max_check=4096)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    recall = np.mean(recalls) * 100
    
    print(f"\nResults:")
    print(f"  Clusters: {index.num_clusters}")
    print(f"  Build time: {build_time:.2f}s")
    print(f"  Latency: p50={p50:.2f}ms, p90={p90:.2f}ms")
    print(f"  Recall@10: {recall:.1f}%")
    
    # Cleanup
    import shutil
    shutil.rmtree(f'./test_{name.replace(" ", "_")}', ignore_errors=True)
    
    return {
        'name': name,
        'clusters': index.num_clusters,
        'build_time': build_time,
        'p50': p50,
        'p90': p90,
        'recall': recall
    }

# Test 1: Baseline (old way)
r1 = test_config("Baseline (target_posting=1000)", 1000, use_ratio=False)

# Test 2: SPTAG-style (ratio-based)
r2 = test_config("SPTAG-style (ratio=0.01)", 118, use_ratio=True)

# Test 3: SPTAG-style NO truncation
r3 = test_config("SPTAG-style NO truncation", 10000, use_ratio=True)

# Summary
print(f"\n{'='*80}")
print("COMPARISON")
print("="*80)
print(f"{'Config':<30} {'Clusters':<10} {'p90(ms)':<10} {'Recall%':<10}")
print("-"*80)
print(f"{r1['name']:<30} {r1['clusters']:<10} {r1['p90']:<10.2f} {r1['recall']:<10.1f}")
print(f"{r2['name']:<30} {r2['clusters']:<10} {r2['p90']:<10.2f} {r2['recall']:<10.1f}")
print(f"{r3['name']:<30} {r3['clusters']:<10} {r3['p90']:<10.2f} {r3['recall']:<10.1f}")
print("="*80)

best = max([r1, r2, r3], key=lambda x: x['recall'])
print(f"\n✅ Best recall: {best['name']} with {best['recall']:.1f}%")
