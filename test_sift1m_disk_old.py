"""
Test SIFT1M with disk-based SPANN
Saves posting lists to disk, loads on-demand during search
"""
import numpy as np
import struct
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk import SPANNDiskBased

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
print("SIFT1M: Disk-Based SPANN Test")
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
    ('1-bit', 1),
    ('2-bit', 2),
    ('4-bit', 4),
]

results = []

for name, bq in configs:
    print("\n" + "="*80)
    print(f"TEST: {name}")
    print("="*80)
    
    disk_path = f'./sift1m_index_{name}'
    
    # Build or load index
    if os.path.exists(disk_path):
        print(f"Loading existing index from {disk_path}...")
        t0 = time.time()
        index = SPANNDiskBased.load(disk_path)
        load_time = time.time() - t0
        print(f"✓ Index loaded in {load_time:.2f}s")
    else:
        print(f"Building new index...")
        t0 = time.time()
        index = SPANNDiskBased(
            dim=128,
            target_posting_size=5000,
            replica_count=6,
            bq=bq,
            use_rabitq=True,
            tree_type='KDT',  # Use KDTree for speed
            disk_path=disk_path
        )
        index.build(base)
        build_time = time.time() - t0
        print(f"✓ Index built in {build_time:.2f}s")
        t0 = time.time()
        index = SPANNDiskBased(
            dim=128,
            target_posting_size=5000,
            replica_count=6,
            bq=bq,
            metric='L2',
            disk_path=disk_path
        )
        index.build(base)
        build_time = time.time() - t0
        print(f"✓ Index built in {build_time:.2f}s")
    
    # Search
    print("\nSearching...")
    t0 = time.time()
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries):
        t_query = time.time()
        dists, indices = index.search(query, base, k=10, search_internal_result_num=128, max_check=4000)
        latency = (time.time() - t_query) * 1000
        latencies.append(latency)
        
        # Compute recall
        gt = set(groundtruth[i])
        found = set(indices[:10])
        recall = len(gt & found) / 10
        recalls.append(recall)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/100 queries")
    
    search_time = time.time() - t0
    
    # Stats
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    qps = len(queries) / search_time
    avg_recall = np.mean(recalls)
    
    print(f"\nResults:")
    print(f"  Search time: {search_time:.2f}s, QPS: {qps:.1f}")
    print(f"  Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
    print(f"  Recall@10: {avg_recall:.1%}")
    
    results.append({
        'name': name,
        'bq': bq,
        'search_time': search_time,
        'qps': qps,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': avg_recall
    })

# Summary
print("\n" + "="*80)
print("SUMMARY: Disk-Based SPANN on SIFT1M")
print("="*80)
print(f"\nConfig     QPS      p50(ms)  p90(ms)  p99(ms)  Recall@10")
print("-"*80)
for r in results:
    print(f"{r['name']:10} {r['qps']:8.1f} {r['p50']:8.2f} {r['p90']:8.2f} {r['p99']:8.2f} {r['recall']:9.1%}")
print("="*80)

print("\nDisk usage:")
for name, bq in configs:
    disk_path = f'./sift1m_index_{name}'
    if os.path.exists(disk_path):
        size = sum(os.path.getsize(os.path.join(dirpath, filename))
                   for dirpath, dirnames, filenames in os.walk(disk_path)
                   for filename in filenames)
        print(f"  {name}: {size/1024**2:.2f} MB")
