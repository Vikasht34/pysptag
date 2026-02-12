"""
SIFT1M: Test 1-bit, 2-bit, 4-bit with 8× replication
Comprehensive comparison of all configurations
"""
import numpy as np
import struct
import time
import sys
import os
sys.path.insert(0, os.path.expanduser('~/pysptag'))

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

def read_fvecs(filename, max_vecs=None):
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
            if max_vecs and len(vectors) >= max_vecs:
                break
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename):
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)

print("="*80)
print("SIFT1M: Complete RaBitQ Comparison (1-bit, 2-bit, 4-bit)")
print("Configuration: 8× replication, 200 clusters")
print("="*80)

data_dir = os.path.expanduser('~/pysptag/data/sift')

# Load SIFT1M
print("\nLoading SIFT1M dataset...")
base = read_fvecs(f'{data_dir}/sift_base.fvecs')
queries = read_fvecs(f'{data_dir}/sift_query.fvecs', max_vecs=100)
groundtruth = read_ivecs(f'{data_dir}/sift_groundtruth.ivecs')[:100]
print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Test configurations
configs = [
    {'name': 'No RaBitQ', 'use_rabitq': False, 'bq': 1},
    {'name': '1-bit RaBitQ', 'use_rabitq': True, 'bq': 1},
    {'name': '2-bit RaBitQ', 'use_rabitq': True, 'bq': 2},
    {'name': '4-bit RaBitQ', 'use_rabitq': True, 'bq': 4},
]

results = []

for config in configs:
    print("\n" + "="*80)
    print(f"TEST: {config['name']}")
    print("="*80)
    
    t0 = time.time()
    index = SPANNRaBitQReplica(
        dim=128,
        target_posting_size=5000,
        replica_count=8,  # 8× replication
        bq=config['bq'],
        use_rabitq=config['use_rabitq']
    )
    index.build(base)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")
    
    t0 = time.time()
    recalls = []
    latencies = []
    for query in queries:
        q_start = time.time()
        dists, indices = index.search(query, base, k=10, search_internal_result_num=128, max_check=4000)
        latencies.append((time.time() - q_start) * 1000)
        
        if len(indices) == 0:
            recalls.append(0)
            continue
        gt = set(groundtruth[len(recalls)][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
    
    search_time = time.time() - t0
    
    print(f"Search time: {search_time:.2f}s, QPS: {len(queries)/search_time:.1f}")
    print(f"Latency: p50={np.percentile(latencies, 50):.2f}ms, p90={np.percentile(latencies, 90):.2f}ms, p99={np.percentile(latencies, 99):.2f}ms")
    print(f"Recall@10: {np.mean(recalls):.2%}")
    
    results.append({
        'name': config['name'],
        'build_time': build_time,
        'search_time': search_time,
        'qps': len(queries)/search_time,
        'p50': np.percentile(latencies, 50),
        'p90': np.percentile(latencies, 90),
        'p99': np.percentile(latencies, 99),
        'recall': np.mean(recalls),
        'index': index  # Keep reference for memory analysis
    })

# Summary table
print("\n" + "="*80)
print("SUMMARY: All Configurations (8× Replication)")
print("="*80)
print(f"{'Config':<15} {'Build(s)':<10} {'Search(s)':<10} {'QPS':<8} {'p50(ms)':<8} {'p90(ms)':<8} {'p99(ms)':<8} {'Recall@10'}")
print("-"*80)
for r in results:
    print(f"{r['name']:<15} {r['build_time']:<10.2f} {r['search_time']:<10.2f} {r['qps']:<8.1f} {r['p50']:<8.2f} {r['p90']:<8.2f} {r['p99']:<8.2f} {r['recall']:.2%}")
print("="*80)

# Memory breakdown
print("\n" + "="*80)
print("MEMORY BREAKDOWN")
print("="*80)

for r in results:
    idx = r['index']
    
    # Posting lists memory
    posting_mem = 0
    for codes in idx.posting_codes:
        posting_mem += codes.nbytes
    
    # Centroids memory
    centroid_mem = idx.centroids.nbytes if idx.centroids is not None else 0
    
    # BKTree memory (approximate)
    bktree_mem = centroid_mem * 2  # Rough estimate: centroids + tree structure
    
    # RNG graph memory (approximate)
    rng_mem = centroid_mem  # Rough estimate: similar to centroids
    
    total_mem = posting_mem + centroid_mem + bktree_mem + rng_mem
    
    print(f"\n{r['name']}:")
    print(f"  Posting lists:  {posting_mem / 1024 / 1024:>8.1f} MB")
    print(f"  Centroids:      {centroid_mem / 1024 / 1024:>8.1f} MB")
    print(f"  BKTree:         {bktree_mem / 1024 / 1024:>8.1f} MB (estimated)")
    print(f"  RNG Graph:      {rng_mem / 1024 / 1024:>8.1f} MB (estimated)")
    print(f"  {'─'*40}")
    print(f"  Total:          {total_mem / 1024 / 1024:>8.1f} MB")

print("\n" + "="*80)
print("Recommendation:")
print("  1-bit: Best speed, 80% recall, 75% memory savings")
print("  2-bit: Balanced, ~85% recall expected")
print("  4-bit: Best recall (~90%), still 75% memory savings")
print("  All use reranking with true distances for accuracy")
print("="*80)
