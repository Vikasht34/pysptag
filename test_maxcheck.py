"""
Test impact of max_check on latency and recall
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("Impact of max_check on Latency and Recall")
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

# Build index
print("Building index...")
index = ProfiledSPANN(
    dim=768,
    target_posting_size=500,
    replica_count=6,
    bq=4,
    use_rabitq=True,
    tree_type='KDT',
    disk_path='./test_maxcheck',
    cache_size=128
)
index.build(data)

# Test different max_check values
max_check_values = [512, 1024, 2048, 4096, 8192]
results = []

for max_check in max_check_values:
    print(f"\n{'='*80}")
    print(f"Testing max_check={max_check}")
    print("="*80)
    
    index.reset_stats()
    recalls = []
    latencies = []
    
    # Warm up cache
    for query in queries[:10]:
        index.search(query, data, k=10, max_check=max_check)
    
    # Actual test
    index.reset_stats()
    for i, query in enumerate(queries[:50]):
        t0 = time.time()
        dists, indices = index.search(query, data, k=10, max_check=max_check)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(gt_indices[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    avg_recall = np.mean(recalls) * 100
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    
    avg_candidates = np.mean(index.stats['num_candidates'])
    avg_rerank_time = np.mean(index.stats['rerank_time'])
    
    print(f"Latency: p50={p50_latency:.2f}ms, p90={p90_latency:.2f}ms")
    print(f"Recall@10: {avg_recall:.1f}%")
    print(f"Avg candidates: {avg_candidates:.0f}")
    print(f"Avg rerank time: {avg_rerank_time:.2f}ms ({avg_rerank_time/p50_latency*100:.1f}% of total)")
    
    results.append({
        'max_check': max_check,
        'p50': p50_latency,
        'p90': p90_latency,
        'recall': avg_recall,
        'candidates': avg_candidates,
        'rerank_time': avg_rerank_time
    })

# Summary
print("\n" + "="*80)
print("SUMMARY: max_check Impact")
print("="*80)
print(f"{'max_check':<12} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10} {'Candidates':<12} {'Rerank(ms)':<12}")
print("-"*80)
for r in results:
    print(f"{r['max_check']:<12} {r['p50']:<10.2f} {r['p90']:<10.2f} {r['recall']:<10.1f} {r['candidates']:<12.0f} {r['rerank_time']:<12.2f}")
print("="*80)

print("\nRecommendation:")
best = min(results, key=lambda x: x['p50'] if x['recall'] > 60 else 999)
print(f"  Use max_check={best['max_check']} for best latency ({best['p50']:.2f}ms) with {best['recall']:.1f}% recall")

# Cleanup
import shutil
shutil.rmtree('./test_maxcheck', ignore_errors=True)
