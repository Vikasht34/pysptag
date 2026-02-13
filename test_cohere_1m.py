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
print("Cohere 1M: Quantization Comparison (1-bit, 2-bit, 4-bit, no-quant)")
print("="*80)

data_file = os.path.expanduser('~/pysptag/data/documents-1m.hdf5')

# Load Cohere 1M from HDF5
print("\nLoading Cohere 1M dataset...")
with h5py.File(data_file, 'r') as f:
    print(f"Keys in file: {list(f.keys())}")
    
    # Check structure
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: Dataset, shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"  {key}: Group, contains={list(item.keys())}")
    
    # Load data based on actual structure
    if 'train' in f and isinstance(f['train'], h5py.Dataset):
        base = f['train'][:]
        queries = f['test'][:100]
        groundtruth = f['neighbors'][:100]
    else:
        # Try alternative structure
        print("\nTrying alternative structure...")
        raise ValueError("Unknown HDF5 structure")

print(f"✓ Base: {base.shape}, Queries: {queries.shape}, GT: {groundtruth.shape}")

# Test each quantization level + no-quant
results = []

for config in [('1-bit', 1, True), ('2-bit', 2, True), ('4-bit', 4, True), ('no-quant', 4, False)]:
    name, bq, use_rabitq = config
    print("\n" + "="*80)
    print(f"TEST: {name}")
    print("="*80)
    
    # Build
    t0 = time.time()
    index = SPANNRaBitQReplica(
        dim=768,
        target_posting_size=5000,
        replica_count=6,
        bq=bq,
        use_rabitq=use_rabitq,
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
print("SUMMARY: Quantization Comparison (Inner Product)")
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
