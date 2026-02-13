"""
Comprehensive test to optimize for p90 <10ms with 90%+ recall.
Focus on posting limits and search parameters.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("P90 Latency Optimization Test")
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

# Test configurations
configs = [
    # (posting_size, replica, search_centroids, max_check, use_limits)
    (118, 8, 32, 4096, True),   # SPTAG-like with limits, fewer centroids
    (118, 8, 48, 4096, True),   # More centroids
    (118, 8, 64, 4096, True),   # Even more centroids
    (118, 6, 64, 4096, True),   # Fewer replicas
    (200, 8, 64, 4096, True),   # Larger postings
]

results = []

for i, (posting_size, replica, search_centroids, max_check, use_limits) in enumerate(configs):
    print(f"\n{'='*80}")
    print(f"Config {i+1}: posting={posting_size}, replica={replica}, "
          f"search_centroids={search_centroids}, max_check={max_check}, limits={use_limits}")
    print("="*80)
    
    # Build index
    index = ProfiledSPANN(
        dim=768,
        target_posting_size=posting_size,
        replica_count=replica,
        bq=4,
        use_rabitq=False,  # No quantization
        tree_type='KDT',
        disk_path=f'./test_config_{i}',
        cache_size=256
    )
    
    t0 = time.time()
    index.build(data, use_posting_limits=use_limits)
    build_time = time.time() - t0
    
    # Warm up
    for query in queries[:10]:
        index.search(query, data, k=10, 
                    search_internal_result_num=search_centroids,
                    max_check=max_check)
    
    # Test
    index.reset_stats()
    recalls = []
    latencies = []
    
    for j, query in enumerate(queries):
        t0 = time.time()
        dists, indices_result = index.search(query, data, k=10, 
                                             search_internal_result_num=search_centroids,
                                             max_check=max_check)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[j][:10])
        found = set(indices_result[:10])
        recalls.append(len(gt & found) / 10)
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    
    # Get stats
    avg_postings = np.mean(index.stats['num_postings_touched'])
    avg_vectors = np.mean(index.stats['num_vectors_in_postings'])
    avg_data_mb = np.mean([b / (1024**2) for b in index.stats['posting_sizes_bytes']])
    
    print(f"\nResults:")
    print(f"  Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
    print(f"  Recall: {avg_recall:.1f}%")
    print(f"  Clusters: {index.num_clusters}")
    print(f"  Postings touched: {avg_postings:.1f}")
    print(f"  Vectors searched: {avg_vectors:.0f}")
    print(f"  Data loaded: {avg_data_mb:.2f} MB/query")
    print(f"  Build time: {build_time:.2f}s")
    
    results.append({
        'config': f"p={posting_size},r={replica},s={search_centroids},m={max_check}",
        'posting_size': posting_size,
        'replica': replica,
        'search_centroids': search_centroids,
        'max_check': max_check,
        'use_limits': use_limits,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': avg_recall,
        'clusters': index.num_clusters,
        'postings': avg_postings,
        'vectors': avg_vectors,
        'data_mb': avg_data_mb,
        'build_time': build_time
    })
    
    # Cleanup
    import shutil
    shutil.rmtree(f'./test_config_{i}', ignore_errors=True)

# Summary
print(f"\n{'='*80}")
print("SUMMARY: All Configurations")
print("="*80)
print(f"{'Config':<40} {'p90(ms)':<10} {'Recall%':<10} {'Vectors':<10} {'Status':<15}")
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
    
    print(f"{r['config']:<40} {r['p90']:<10.2f} {r['recall']:<10.1f} {r['vectors']:<10.0f} {status:<15}")

print("="*80)

# Find best config
best = min([r for r in results if r['recall'] >= 90], 
           key=lambda x: x['p90'], default=None)

if best:
    print(f"\n✅ BEST CONFIG:")
    print(f"  posting_size={best['posting_size']}, replica={best['replica']}, "
          f"search_centroids={best['search_centroids']}, max_check={best['max_check']}")
    print(f"  p90={best['p90']:.2f}ms, recall={best['recall']:.1f}%")
    print(f"  Searches {best['vectors']:.0f} vectors across {best['postings']:.0f} postings")
else:
    print(f"\n❌ No config met target (p90 <10ms with 90%+ recall)")
    print(f"   Need to optimize further!")
