"""
Compare BKTree vs KDTree for SPANN centroid search
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.expanduser('~/pysptag'))

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

print("="*80)
print("BKTree vs KDTree Comparison")
print("="*80)

# Generate test data
np.random.seed(42)
data = np.random.randn(10000, 128).astype(np.float32)
queries = np.random.randn(100, 128).astype(np.float32)

# Ground truth (brute force)
print("\nComputing ground truth...")
gt_indices = []
for query in queries:
    dists = np.sum((data - query) ** 2, axis=1)
    gt_indices.append(np.argsort(dists)[:10])
gt_indices = np.array(gt_indices)

results = []

for tree_type in ['BKT', 'KDT']:
    print(f"\n{'='*80}")
    print(f"Testing {tree_type}")
    print("="*80)
    
    # Build index
    t0 = time.time()
    index = SPANNRaBitQReplica(
        dim=128,
        target_posting_size=500,
        replica_count=6,
        bq=4,
        use_rabitq=True,
        tree_type=tree_type
    )
    index.build(data)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")
    
    # Search
    t0 = time.time()
    recalls = []
    latencies = []
    for i, query in enumerate(queries):
        q_start = time.time()
        dists, indices = index.search(query, data, k=10, search_internal_result_num=64)
        latencies.append((time.time() - q_start) * 1000)
        
        if len(indices) == 0:
            recalls.append(0)
            continue
            
        gt = set(gt_indices[i])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    search_time = time.time() - t0
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    qps = len(queries) / search_time
    
    print(f"Search time: {search_time:.2f}s, QPS: {qps:.1f}")
    print(f"Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
    print(f"Recall@10: {avg_recall:.2f}%")
    
    results.append({
        'tree': tree_type,
        'build': build_time,
        'search': search_time,
        'qps': qps,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': avg_recall
    })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Tree':<8} {'Build(s)':<10} {'Search(s)':<10} {'QPS':<8} {'p50(ms)':<8} {'p90(ms)':<8} {'p99(ms)':<8} {'Recall':<8}")
print("-"*80)
for r in results:
    print(f"{r['tree']:<8} {r['build']:<10.2f} {r['search']:<10.2f} {r['qps']:<8.1f} {r['p50']:<8.2f} {r['p90']:<8.2f} {r['p99']:<8.2f} {r['recall']:<8.2f}%")
print("="*80)
