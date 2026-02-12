"""
SIFT1M Comparison: SPANN with vs without RaBitQ
Apple-to-apple comparison on full 1M dataset
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
print("SIFT1M Comparison: SPANN with vs without RaBitQ")
print("="*80)

data_dir = os.path.expanduser('~/pysptag/data/sift')

# Load full SIFT1M
print("\nLoading SIFT1M dataset...")
base = read_fvecs(f'{data_dir}/sift_base.fvecs')
queries = read_fvecs(f'{data_dir}/sift_query.fvecs', max_vecs=100)
groundtruth = read_ivecs(f'{data_dir}/sift_groundtruth.ivecs')[:100]
print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Test 1: No RaBitQ
print("\n" + "="*80)
print("TEST 1: SPANN (No Quantization)")
print("="*80)

t0 = time.time()
index1 = SPANNRaBitQReplica(
    dim=128,
    target_posting_size=5000,
    replica_count=2,
    use_rabitq=False
)
index1.build(base)
build_time1 = time.time() - t0
print(f"Build time: {build_time1:.2f}s")

t0 = time.time()
recalls1 = []
for query in queries:
    dists, indices = index1.search(query, base, k=10, search_internal_result_num=64, max_check=2000)
    if len(indices) == 0:
        recalls1.append(0)
        continue
    gt = set(groundtruth[len(recalls1)][:10])
    found = set(indices[:10])
    recalls1.append(len(gt & found) / 10)

search_time1 = time.time() - t0
print(f"Search time: {search_time1:.2f}s, QPS: {len(queries)/search_time1:.1f}")
print(f"Recall@10: {np.mean(recalls1):.2%}")

# Test 2: With RaBitQ
print("\n" + "="*80)
print("TEST 2: SPANN + RaBitQ")
print("="*80)

t0 = time.time()
index2 = SPANNRaBitQReplica(
    dim=128,
    target_posting_size=5000,
    replica_count=2,
    use_rabitq=True
)
index2.build(base)
build_time2 = time.time() - t0
print(f"Build time: {build_time2:.2f}s")

t0 = time.time()
recalls2 = []
for query in queries:
    dists, indices = index2.search(query, base, k=10, search_internal_result_num=64, max_check=2000)
    if len(indices) == 0:
        recalls2.append(0)
        continue
    gt = set(groundtruth[len(recalls2)][:10])
    found = set(indices[:10])
    recalls2.append(len(gt & found) / 10)

search_time2 = time.time() - t0
print(f"Search time: {search_time2:.2f}s, QPS: {len(queries)/search_time2:.1f}")
print(f"Recall@10: {np.mean(recalls2):.2%}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Metric':<20} {'No RaBitQ':<15} {'With RaBitQ':<15} {'Difference'}")
print("-"*80)
print(f"{'Build Time (s)':<20} {build_time1:<15.2f} {build_time2:<15.2f} {build_time2-build_time1:+.2f}s")
print(f"{'Search Time (s)':<20} {search_time1:<15.2f} {search_time2:<15.2f} {search_time2-search_time1:+.2f}s")
print(f"{'QPS':<20} {len(queries)/search_time1:<15.1f} {len(queries)/search_time2:<15.1f} {len(queries)/search_time2-len(queries)/search_time1:+.1f}")
print(f"{'Recall@10':<20} {np.mean(recalls1):<15.2%} {np.mean(recalls2):<15.2%} {np.mean(recalls2)-np.mean(recalls1):+.2%}")
print("="*80)
