"""
Cohere 1M: Test all quantization levels (1-bit, 2-bit, 4-bit)
Dataset: 1M vectors, 768D, Inner Product metric
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

def read_ivecs(filename, max_vecs=None):
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
            if max_vecs and len(vectors) >= max_vecs:
                break
    return np.array(vectors, dtype=np.int32)

print("="*80)
print("Cohere 1M: RaBitQ Quantization Comparison (Inner Product)")
print("="*80)

data_dir = os.path.expanduser('~/pysptag/data/cohere')

# Load Cohere 1M
print("\nLoading Cohere 1M dataset...")
base = read_fvecs(f'{data_dir}/base.1M.fvecs')
queries = read_fvecs(f'{data_dir}/queries.dev.fvecs', max_vecs=100)
groundtruth = read_ivecs(f'{data_dir}/gt_dev.ivecs', max_vecs=100)
print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Test each quantization level
for bq in [1, 2, 4]:
    print("\n" + "="*80)
    print(f"TEST: {bq}-bit Quantization")
    print("="*80)
    
    # Build
    t0 = time.time()
    index = SPANNRaBitQReplica(
        dim=768,
        target_posting_size=5000,
        replica_count=6,
        bq=bq,
        use_rabitq=True,
        metric='IP'
    )
    index.build(base)
    build_time = time.time() - t0
    print(f"Build time: {build_time:.2f}s")
    
    # Search
    t0 = time.time()
    recalls = []
    latencies = []
    for i, query in enumerate(queries):
        if i % 10 == 0:
            print(f"  Query {i}/100...")
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
    
    print(f"\nSearch time: {search_time:.2f}s, QPS: {len(queries)/search_time:.1f}")
    print(f"Latency: p50={np.percentile(latencies, 50):.2f}ms, p90={np.percentile(latencies, 90):.2f}ms, p99={np.percentile(latencies, 99):.2f}ms")
    print(f"Recall@10: {np.mean(recalls):.2%}")

print("\n" + "="*80)
print("DONE: Cohere 1M with Inner Product metric")
print("="*80)
