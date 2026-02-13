"""
Test SPTAG config with higher max_check for better recall
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("SPTAG Config with Higher max_check")
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

# Build index once
print("\nBuilding index...")
index = ProfiledSPANN(
    dim=768,
    target_posting_size=118,
    replica_count=8,
    bq=4,
    use_rabitq=False,  # No quantization
    tree_type='KDT',
    disk_path='./test_maxcheck_sweep',
    cache_size=256
)
index.build(data)
print(f"✓ Built: {index.num_clusters} clusters")

# Test different max_check values
max_check_values = [2048, 4096, 8192, 16384]
results = []

for max_check in max_check_values:
    print(f"\n{'='*80}")
    print(f"Testing max_check={max_check}")
    print("="*80)
    
    # Warm up
    for query in queries[:5]:
        index.search(query, data, k=10, search_internal_result_num=64, max_check=max_check)
    
    # Test
    index.reset_stats()
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries[:50]):
        t0 = time.time()
        dists, indices = index.search(query, data, k=10, 
                                      search_internal_result_num=64,
                                      max_check=max_check)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    
    avg_search = np.mean(index.stats['posting_search_time'])
    avg_rerank = np.mean(index.stats['rerank_time'])
    
    print(f"Latency: p50={p50:.2f}ms, p90={p90:.2f}ms")
    print(f"Recall@10: {avg_recall:.1f}%")
    print(f"Posting search: {avg_search:.2f}ms ({avg_search/p50*100:.1f}%)")
    print(f"Reranking: {avg_rerank:.2f}ms ({avg_rerank/p50*100:.1f}%)")
    
    results.append({
        'max_check': max_check,
        'p50': p50,
        'p90': p90,
        'recall': avg_recall,
        'search_ms': avg_search,
        'rerank_ms': avg_rerank
    })

# Summary
print(f"\n{'='*80}")
print("SUMMARY: max_check Impact (No Quantization)")
print("="*80)
print(f"{'max_check':<12} {'p50(ms)':<10} {'Recall%':<10} {'Search(ms)':<12} {'Rerank(ms)':<12}")
print("-"*80)
for r in results:
    print(f"{r['max_check']:<12} {r['p50']:<10.2f} {r['recall']:<10.1f} {r['search_ms']:<12.2f} {r['rerank_ms']:<12.2f}")
print("="*80)

# Find sweet spot
print("\nAnalysis:")
for r in results:
    if r['recall'] > 85 and r['p50'] < 10:
        print(f"✅ max_check={r['max_check']}: {r['p50']:.2f}ms with {r['recall']:.1f}% recall - MEETS TARGET!")
    elif r['recall'] > 85:
        print(f"⚠️  max_check={r['max_check']}: {r['recall']:.1f}% recall but {r['p50']:.2f}ms (too slow)")
    elif r['p50'] < 10:
        print(f"⚠️  max_check={r['max_check']}: {r['p50']:.2f}ms but only {r['recall']:.1f}% recall")

# Cleanup
import shutil
shutil.rmtree('./test_maxcheck_sweep', ignore_errors=True)
