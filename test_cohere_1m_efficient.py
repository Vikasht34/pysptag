"""
Cohere 1M: Efficient test - cluster once, quantize multiple times
Compares 1-bit, 2-bit, 4-bit, no-quant using same clustering
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
print("Cohere 1M: Efficient Quantization Comparison (shared clustering)")
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

# Build base index with clustering (use 4-bit as base)
print("\n" + "="*80)
print("STEP 1: Build base index with clustering (one-time cost)")
print("="*80)

t0 = time.time()
base_index = SPANNRaBitQReplica(
    dim=768,
    target_posting_size=5000,
    replica_count=6,
    bq=4,
    use_rabitq=True,
    metric='IP'
)
base_index.build(base)
cluster_time = time.time() - t0
print(f"✓ Clustering + base build: {cluster_time:.2f}s")

# Extract clustering info
centroids = base_index.centroids
posting_lists = base_index.posting_lists
print(f"✓ Centroids: {centroids.shape}, Postings: {len(posting_lists)}")

# Test each quantization using same clustering
results = []

for config in [('1-bit', 1, True), ('2-bit', 2, True), ('4-bit', 4, True), ('no-quant', 4, False)]:
    name, bq, use_rabitq = config
    print("\n" + "="*80)
    print(f"TEST: {name} (reusing clustering)")
    print("="*80)
    
    # Create index with same clustering
    t0 = time.time()
    index = SPANNRaBitQReplica(
        dim=768,
        target_posting_size=5000,
        replica_count=6,
        bq=bq,
        use_rabitq=use_rabitq,
        metric='IP'
    )
    
    # Reuse clustering from base_index
    index.centroids = centroids
    index.posting_lists = posting_lists
    index.num_clusters = len(posting_lists)
    
    # Only quantize posting lists (skip clustering!)
    print("Quantizing posting lists...")
    index.posting_codes = []
    index.posting_rabitqs = []
    
    for i, posting_ids in enumerate(posting_lists):
        if len(posting_ids) == 0:
            index.posting_rabitqs.append(None)
            index.posting_codes.append(np.array([]))
            continue
        
        posting_vecs = base[posting_ids]
        
        if use_rabitq:
            from src.quantization.rabitq import RaBitQ
            rabitq = RaBitQ(dim=768, bq=bq, metric='IP')
            codes = rabitq.build(posting_vecs)
            index.posting_rabitqs.append(rabitq)
            index.posting_codes.append(codes)
        else:
            # No quantization - store original vectors
            index.posting_rabitqs.append(None)
            index.posting_codes.append(posting_vecs)
    
    # Build BKTree+RNG on centroids
    from src.core.bktree import BKTree
    from src.core.rng import RNG
    index.bktree = BKTree(num_trees=1, kmeans_k=32)
    index.bktree.build(centroids)
    index.rng = RNG(neighborhood_size=32, metric='IP')
    index.rng.build(centroids)
    
    build_time = time.time() - t0
    print(f"Quantization time: {build_time:.2f}s (vs {cluster_time:.2f}s full build)")
    
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
print("SUMMARY: Quantization Comparison (shared clustering, Inner Product)")
print("="*80)
print(f"Clustering time: {cluster_time:.2f}s (one-time cost)")
print()
print(f"{'Config':<10} {'Quant(s)':<10} {'Search(s)':<10} {'QPS':<8} {'p50(ms)':<8} {'p90(ms)':<8} {'p99(ms)':<8} {'Recall@10'}")
print("-"*80)
for r in results:
    print(f"{r['name']:<10} {r['build_time']:<10.2f} {r['search_time']:<10.2f} {r['qps']:<8.1f} {r['p50']:<8.2f} {r['p90']:<8.2f} {r['p99']:<8.2f} {r['recall']:.2%}")
print("="*80)

total_time = cluster_time + sum(r['build_time'] for r in results)
print(f"\nTotal time: {total_time:.2f}s (vs {cluster_time * 4:.2f}s if clustering 4 times)")
print(f"Time saved: {cluster_time * 3:.2f}s ({cluster_time * 3 / (cluster_time * 4) * 100:.0f}% faster)")

print("\nRecommendation:")
print("  1-bit: Fastest, 32× compression - use for very large scale")
print("  2-bit: Balanced, 16× compression - good middle ground")
print("  4-bit: Best recall, 8× compression - recommended for production")
print("  no-quant: Highest recall, no compression - baseline comparison")
print("="*80)
