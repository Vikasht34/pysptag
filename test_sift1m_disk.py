"""
Test SPTAG-optimized SPANN on SIFT1M
Build once, test multiple quantization levels
Target: p90 <10ms with 90%+ recall on EBS
"""
import numpy as np
import time
import os

print("="*80)
print("SIFT1M Disk-Based Test (SPTAG-Optimized)")
print("="*80)

# Load SIFT1M
print("\nLoading SIFT1M...")
base = np.fromfile('data/sift/sift_base.fvecs', dtype=np.float32)
base = base.reshape(-1, 129)[:, 1:].copy()
print(f"  Base: {base.shape}")

queries = np.fromfile('data/sift/sift_query.fvecs', dtype=np.float32)
queries = queries.reshape(-1, 129)[:, 1:].copy()
print(f"  Queries: {queries.shape}")

groundtruth = np.fromfile('data/sift/sift_groundtruth.ivecs', dtype=np.int32)
groundtruth = groundtruth.reshape(-1, 101)[:, 1:].copy()

# Build index ONCE
print("\nBuilding index (clustering only, done once)...")
print("  Config: ratio=0.01 (~10K clusters), replica=8, posting_limit=500")

from src.index.spann_disk_optimized import SPANNDiskOptimized

index = SPANNDiskOptimized(
    dim=128,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    tree_type='KDT',
    disk_path='./sift1m_disk_sptag',
    cache_size=1024,
    clustering='kmeans'
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")
print(f"✓ Clusters: {index.num_clusters}")

# Test different quantization levels
configs = [
    ('no-quant', False, 4),
    ('1-bit', True, 1),
    ('2-bit', True, 2),
    ('4-bit', True, 4),
]

results = []

for name, use_rabitq, bq in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print("="*80)
    
    # Update quantization
    index.use_rabitq = use_rabitq
    index.bq = bq
    if use_rabitq:
        from src.quantization.rabitq_numba import RaBitQNumba
        index.rabitq = RaBitQNumba(dim=128, bq=bq)
        index.rabitq.train(base[:10000])
    
    # Warm up
    print("Warming up...")
    for q in queries[:20]:
        index.search(q, base, k=10, search_internal_result_num=64, max_check=8192)
    
    # Test
    print("Testing (100 queries)...")
    recalls = []
    latencies = []
    
    for i, q in enumerate(queries[:100]):
        t0 = time.time()
        dists, indices = index.search(q, base, k=10, 
                                      search_internal_result_num=64,
                                      max_check=8192)
        latencies.append((time.time() - t0) * 1000)
        
        gt = set(groundtruth[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100...")
    
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    recall = np.mean(recalls) * 100
    
    print(f"Results: p50={p50:.2f}ms, p90={p90:.2f}ms, recall={recall:.1f}%")
    
    results.append({
        'config': name,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': recall
    })

# Summary
print(f"\n{'='*80}")
print("SUMMARY: SIFT1M")
print("="*80)
print(f"Build: {build_time:.1f}s, Clusters: {index.num_clusters}")
print()
print(f"{'Config':<12} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10} {'Status':<8}")
print("-"*80)
for r in results:
    status = '✅' if r['p90'] < 10 and r['recall'] >= 90 else ''
    print(f"{r['config']:<12} {r['p50']:<10.2f} {r['p90']:<10.2f} {r['recall']:<10.1f} {status}")
print("="*80)

best = max(results, key=lambda x: x['recall'] if x['p90'] < 10 else 0)
if best['p90'] < 10 and best['recall'] >= 90:
    print(f"\n✅ SUCCESS: {best['config']} - p90={best['p90']:.2f}ms, recall={best['recall']:.1f}%")
else:
    print(f"\n⚠️  Best: {best['config']} - p90={best['p90']:.2f}ms, recall={best['recall']:.1f}%")

print(f"\nCleanup: rm -rf ./sift1m_disk_sptag")
