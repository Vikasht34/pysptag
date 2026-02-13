"""
EC2 Test Script for SPTAG-Optimized Disk-Based SPANN on Cohere 1M
Build index once, test multiple quantization levels

Target: p90 <10ms with 90%+ recall on EBS
"""
import numpy as np
import h5py
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

def get_dir_size(path):
    """Get directory size in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

print("="*80)
print("EC2: SPTAG-Optimized SPANN on Cohere 1M")
print("="*80)

# Configuration
DATA_FILE = os.path.expanduser('~/pysptag/cohere-wikipedia-768-angular.hdf5')
INDEX_DIR = '/mnt/ebs/spann_cohere_sptag'
NUM_QUERIES = 100

# Load Cohere 1M
print("\nLoading Cohere 1M...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Build index ONCE (clustering is expensive)
print("\n" + "="*80)
print("Building Index (clustering only, done once)")
print("="*80)
print("  Config: ratio=0.01 (~10K clusters), replica=8")
print("  Posting limit: 500")

print(f"\nBuilding...")
t0 = time.time()
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='Cosine',
    tree_type='KDT',
    disk_path=INDEX_DIR,
    cache_size=1024,
    clustering='kmeans'
)
index.build(base)
build_time = time.time() - t0
print(f"✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")

disk_size = get_dir_size(INDEX_DIR)
print(f"✓ Clusters: {index.num_clusters}")
print(f"✓ Disk: {disk_size/1024**3:.2f} GB")

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
        index.rabitq = RaBitQNumba(dim=768, bq=bq)
        index.rabitq.train(base[:10000])
    
    # Warm up
    print("Warming up...")
    for q in queries[:20]:
        index.search(q, base, k=10, search_internal_result_num=64, max_check=8192)
    
    # Test
    print(f"Testing ({NUM_QUERIES} queries)...")
    t0 = time.time()
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries):
        q_start = time.time()
        dists, indices = index.search(
            query, base, k=10, 
            search_internal_result_num=64,
            max_check=8192
        )
        latencies.append((time.time() - q_start) * 1000)
        
        if len(indices) == 0:
            recalls.append(0)
            continue
        
        gt = set(groundtruth[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{NUM_QUERIES}...")
    
    search_time = time.time() - t0
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    qps = NUM_QUERIES / search_time
    
    print(f"Results: p50={p50:.2f}ms, p90={p90:.2f}ms, recall={avg_recall:.1f}%")
    
    results.append({
        'config': name,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': avg_recall,
        'qps': qps
    })

# Summary
print(f"\n{'='*80}")
print("SUMMARY: Cohere 1M on EBS")
print("="*80)
print(f"Build: {build_time:.1f}s, Clusters: {index.num_clusters}, Disk: {disk_size/1024**3:.2f} GB")
print()
print(f"{'Config':<12} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10} {'QPS':<8} {'Status':<8}")
print("-"*80)
for r in results:
    status = '✅' if r['p90'] < 10 and r['recall'] >= 90 else ''
    print(f"{r['config']:<12} {r['p50']:<10.2f} {r['p90']:<10.2f} {r['recall']:<10.1f} {r['qps']:<8.1f} {status}")
print("="*80)

best = max(results, key=lambda x: x['recall'] if x['p90'] < 10 else 0)
if best['p90'] < 10 and best['recall'] >= 90:
    print(f"\n✅ SUCCESS: {best['config']} - p90={best['p90']:.2f}ms, recall={best['recall']:.1f}%")
else:
    print(f"\n⚠️  Best: {best['config']} - p90={best['p90']:.2f}ms, recall={best['recall']:.1f}%")

print(f"\nCleanup: rm -rf {INDEX_DIR}")
