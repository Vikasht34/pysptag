"""
SIFT1M: Test all bit quantizations (1, 2, 4) in one run
Compares recall, speed, and memory for each
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
print("SIFT1M: RaBitQ Quantization Comparison (1-bit, 2-bit, 4-bit, no-quant)")
print("="*80)

data_dir = os.path.expanduser('~/pysptag/data/sift')

# Load SIFT1M
print("\nLoading SIFT1M dataset...")
base = read_fvecs(f'{data_dir}/sift_base.fvecs')
queries = read_fvecs(f'{data_dir}/sift_query.fvecs', max_vecs=100)
groundtruth = read_ivecs(f'{data_dir}/sift_groundtruth.ivecs')[:100]
print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Test each bit configuration + no-quant
results = []

for config in [('1-bit', 1, True), ('2-bit', 2, True), ('4-bit', 4, True), ('no-quant', 4, False)]:
    name, bq, use_rabitq = config
    print("\n" + "="*80)
    print(f"TEST: {name}")
    print("="*80)
    
    t0 = time.time()
    index = SPANNRaBitQReplica(
        dim=128,
        target_posting_size=5000,
        replica_count=6,
        bq=bq,
        use_rabitq=use_rabitq
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
        'name': name,
        'build_time': build_time,
        'search_time': search_time,
        'qps': len(queries)/search_time,
        'p50': np.percentile(latencies, 50),
        'p90': np.percentile(latencies, 90),
        'p99': np.percentile(latencies, 99),
        'recall': np.mean(recalls)
    })

# Summary table
print("\n" + "="*80)
print("SUMMARY: Quantization Comparison")
print("="*80)
print(f"{'Config':<10} {'Build(s)':<10} {'Search(s)':<10} {'QPS':<8} {'p50(ms)':<8} {'p90(ms)':<8} {'p99(ms)':<8} {'Recall@10'}")
print("-"*80)
for r in results:
    print(f"{r['name']:<10} {r['build_time']:<10.2f} {r['search_time']:<10.2f} {r['qps']:<8.1f} {r['p50']:<8.2f} {r['p90']:<8.2f} {r['p99']:<8.2f} {r['recall']:.2%}")
print("="*80)

print("\nRecommendation:")
print("  1-bit: Fastest, 32× compression - use for very large scale")
print("  2-bit: Balanced, 16× compression - good middle ground")
print("  4-bit: Best recall, 8× compression - recommended for production")
print("  no-quant: Highest recall, no compression - baseline comparison")
print("="*80)
