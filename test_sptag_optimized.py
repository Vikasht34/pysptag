"""
SPANN with SPTAG-optimized parameters:
- Small posting size (118 vectors like SPTAG)
- Many clusters
- Search fewer centroids
- Target: <5ms latency with 90%+ recall
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("SPTAG-Optimized Configuration Test")
print("="*80)

# Generate test data (10K vectors, 768-dim)
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
    {
        'name': 'Current (baseline)',
        'posting_size': 5000,
        'replica': 6,
        'search_centroids': 20,
        'max_check': 4096
    },
    {
        'name': 'SPTAG-like (118 vectors/posting)',
        'posting_size': 118,
        'replica': 8,
        'search_centroids': 32,
        'max_check': 2048
    },
    {
        'name': 'Aggressive (50 vectors/posting)',
        'posting_size': 50,
        'replica': 8,
        'search_centroids': 48,
        'max_check': 2048
    },
    {
        'name': 'Balanced (200 vectors/posting)',
        'posting_size': 200,
        'replica': 8,
        'search_centroids': 40,
        'max_check': 2048
    },
]

results = []

for config in configs:
    print(f"\n{'='*80}")
    print(f"Config: {config['name']}")
    print(f"  posting_size={config['posting_size']}, replica={config['replica']}")
    print(f"  search_centroids={config['search_centroids']}, max_check={config['max_check']}")
    print("="*80)
    
    # Build index
    index = ProfiledSPANN(
        dim=768,
        target_posting_size=config['posting_size'],
        replica_count=config['replica'],
        bq=4,
        use_rabitq=True,
        tree_type='KDT',
        disk_path=f'./test_{config["name"].replace(" ", "_").replace("/", "_")}',
        cache_size=256
    )
    
    t0 = time.time()
    index.build(data)
    build_time = time.time() - t0
    
    num_clusters = index.num_clusters
    print(f"Built: {num_clusters} clusters, build time: {build_time:.2f}s")
    
    # Warm up cache
    for query in queries[:10]:
        index.search(query, data, k=10, 
                    search_internal_result_num=config['search_centroids'],
                    max_check=config['max_check'])
    
    # Test
    index.reset_stats()
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries[:100]):
        t0 = time.time()
        dists, indices = index.search(query, data, k=10, 
                                      search_internal_result_num=config['search_centroids'],
                                      max_check=config['max_check'])
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    
    avg_postings = np.mean(index.stats['num_postings_touched'])
    avg_vectors = np.mean(index.stats['num_vectors_in_postings'])
    avg_candidates = np.mean(index.stats['num_candidates'])
    
    avg_centroid = np.mean(index.stats['centroid_search_time'])
    avg_load = np.mean(index.stats['posting_load_time'])
    avg_search = np.mean(index.stats['posting_search_time'])
    avg_rerank = np.mean(index.stats['rerank_time'])
    total_avg = avg_centroid + avg_load + avg_search + avg_rerank
    
    print(f"\nResults:")
    print(f"  Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
    print(f"  Recall@10: {avg_recall:.1f}%")
    print(f"  Postings touched: {avg_postings:.0f}")
    print(f"  Vectors searched: {avg_vectors:.0f}")
    print(f"  Candidates reranked: {avg_candidates:.0f}")
    print(f"\nTiming breakdown:")
    print(f"  Centroid: {avg_centroid:.2f}ms ({avg_centroid/total_avg*100:.1f}%)")
    print(f"  Load: {avg_load:.2f}ms ({avg_load/total_avg*100:.1f}%)")
    print(f"  Search: {avg_search:.2f}ms ({avg_search/total_avg*100:.1f}%)")
    print(f"  Rerank: {avg_rerank:.2f}ms ({avg_rerank/total_avg*100:.1f}%)")
    
    index.print_cache_stats()
    
    results.append({
        'name': config['name'],
        'num_clusters': num_clusters,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': avg_recall,
        'vectors': avg_vectors,
        'candidates': avg_candidates,
        'centroid_ms': avg_centroid,
        'load_ms': avg_load,
        'search_ms': avg_search,
        'rerank_ms': avg_rerank
    })

# Summary
print("\n" + "="*80)
print("SUMMARY: SPTAG-Optimized vs Current")
print("="*80)
print(f"{'Config':<30} {'Clusters':<10} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10} {'Vectors':<10}")
print("-"*80)
for r in results:
    print(f"{r['name']:<30} {r['num_clusters']:<10} {r['p50']:<10.2f} {r['p90']:<10.2f} {r['recall']:<10.1f} {r['vectors']:<10.0f}")
print("="*80)

# Find best config
print("\nAnalysis:")
baseline = results[0]
print(f"Baseline: {baseline['p50']:.2f}ms, {baseline['recall']:.1f}% recall, {baseline['vectors']:.0f} vectors")

for r in results[1:]:
    speedup = baseline['p50'] / r['p50']
    vector_reduction = (1 - r['vectors'] / baseline['vectors']) * 100
    recall_change = r['recall'] - baseline['recall']
    
    print(f"\n{r['name']}:")
    print(f"  Speedup: {speedup:.2f}× ({r['p50']:.2f}ms)")
    print(f"  Vectors: {vector_reduction:+.1f}% ({r['vectors']:.0f})")
    print(f"  Recall: {recall_change:+.1f}% ({r['recall']:.1f}%)")
    
    if r['p50'] < 5.0 and r['recall'] > 85:
        print(f"  ✅ MEETS TARGET: <5ms with >85% recall!")

# Cleanup
import shutil
for r in results:
    shutil.rmtree(f'./test_{r["name"].replace(" ", "_").replace("/", "_")}', ignore_errors=True)
