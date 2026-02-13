"""
Test with SPTAG-like parameters: small posting sizes
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("SPTAG-like Configuration: Small Posting Sizes")
print("="*80)

# Generate test data
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

configs = [
    ("Current (bad)", 5000, 6),
    ("SPTAG-like", 118, 6),
    ("Medium", 500, 6),
    ("Small", 200, 6),
]

results = []

for name, posting_size, replica in configs:
    print(f"\n{'='*80}")
    print(f"Config: {name}")
    print(f"  target_posting_size={posting_size}, replica_count={replica}")
    print("="*80)
    
    # Build index
    index = ProfiledSPANN(
        dim=768,
        target_posting_size=posting_size,
        replica_count=replica,
        bq=4,
        use_rabitq=True,
        tree_type='KDT',
        disk_path=f'./test_{name.replace(" ", "_")}',
        cache_size=256
    )
    
    t0 = time.time()
    index.build(data)
    build_time = time.time() - t0
    
    num_clusters = index.num_clusters
    print(f"Built: {num_clusters} clusters, build time: {build_time:.2f}s")
    
    # Warm up cache
    for query in queries[:10]:
        index.search(query, data, k=10, search_internal_result_num=min(64, num_clusters))
    
    # Test
    index.reset_stats()
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries[:50]):
        t0 = time.time()
        dists, indices = index.search(query, data, k=10, 
                                      search_internal_result_num=min(64, num_clusters),
                                      max_check=2048)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    
    avg_postings = np.mean(index.stats['num_postings_touched'])
    avg_vectors = np.mean(index.stats['num_vectors_in_postings'])
    avg_candidates = np.mean(index.stats['num_candidates'])
    avg_data_mb = np.mean(index.stats['total_data_loaded_mb'])
    
    avg_centroid = np.mean(index.stats['centroid_search_time'])
    avg_load = np.mean(index.stats['posting_load_time'])
    avg_search = np.mean(index.stats['posting_search_time'])
    avg_rerank = np.mean(index.stats['rerank_time'])
    
    print(f"\nResults:")
    print(f"  Latency: p50={p50:.2f}ms, p90={p90:.2f}ms")
    print(f"  Recall@10: {avg_recall:.1f}%")
    print(f"  Postings touched: {avg_postings:.0f}")
    print(f"  Vectors in postings: {avg_vectors:.0f}")
    print(f"  Candidates: {avg_candidates:.0f}")
    print(f"  Data loaded: {avg_data_mb:.2f} MB")
    print(f"\nTiming breakdown:")
    print(f"  Centroid: {avg_centroid:.2f}ms ({avg_centroid/p50*100:.1f}%)")
    print(f"  Load: {avg_load:.2f}ms ({avg_load/p50*100:.1f}%)")
    print(f"  Search: {avg_search:.2f}ms ({avg_search/p50*100:.1f}%)")
    print(f"  Rerank: {avg_rerank:.2f}ms ({avg_rerank/p50*100:.1f}%)")
    
    results.append({
        'name': name,
        'posting_size': posting_size,
        'num_clusters': num_clusters,
        'p50': p50,
        'p90': p90,
        'recall': avg_recall,
        'postings': avg_postings,
        'vectors': avg_vectors,
        'candidates': avg_candidates,
        'data_mb': avg_data_mb
    })

# Summary
print("\n" + "="*80)
print("SUMMARY: Impact of Posting Size")
print("="*80)
print(f"{'Config':<20} {'Clusters':<10} {'p50(ms)':<10} {'Recall%':<10} {'Vectors':<10} {'Data(MB)':<10}")
print("-"*80)
for r in results:
    print(f"{r['name']:<20} {r['num_clusters']:<10} {r['p50']:<10.2f} {r['recall']:<10.1f} {r['vectors']:<10.0f} {r['data_mb']:<10.2f}")
print("="*80)

print("\nKey Insight:")
best = min(results, key=lambda x: x['p50'] if x['recall'] > 60 else 999)
print(f"  Best: {best['name']} with {best['num_clusters']} clusters")
print(f"  Latency: {best['p50']:.2f}ms, Recall: {best['recall']:.1f}%")
print(f"  Searches only {best['vectors']:.0f} vectors (vs {results[0]['vectors']:.0f} in current)")

# Cleanup
import shutil
for r in results:
    shutil.rmtree(f'./test_{r["name"].replace(" ", "_")}', ignore_errors=True)
