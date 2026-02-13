"""
Cohere 1M: Test all quantization levels (1-bit, 2-bit, 4-bit)
Dataset: 1M vectors, 768D, Inner Product metric
"""
import numpy as np
import h5py
import time
import sys
import os
sys.path.insert(0, os.path.expanduser('~/pysptag'))

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

print("="*80)
print("Cohere 1M: RaBitQ Quantization Comparison (Inner Product)")
print("="*80)

data_file = os.path.expanduser('~/pysptag/data/documents-1m.hdf5')

# Load Cohere 1M from HDF5
print("\nLoading Cohere 1M dataset...")
with h5py.File(data_file, 'r') as f:
    base = f['train'][:]  # or 'ingest' - use train for building
    queries = f['test'][:100]  # 100 queries
    groundtruth = f['neighbors'][:100]  # Ground truth neighbors
print(f"✓ Base: {base.shape}, Queries: {queries.shape}, GT: {groundtruth.shape}")

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
    
    # Print posting stats
    posting_sizes = [len(p) for p in index.posting_lists]
    print(f"Posting sizes: min={min(posting_sizes)}, max={max(posting_sizes)}, avg={sum(posting_sizes)/len(posting_sizes):.0f}")
    print(f"Build time: {build_time:.2f}s")
    
    # Search
    t0 = time.time()
    recalls = []
    latencies = []
    for i, query in enumerate(queries):
        if i % 10 == 0:
            print(f"  Query {i}/100...")
        q_start = time.time()
        dists, indices = index.search(query, base, k=10, search_internal_result_num=256, max_check=8000)
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
