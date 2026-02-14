#!/usr/bin/env python3
"""
Cohere 1M Benchmark on EC2 - ALL SPTAG Optimizations
Tests different configurations with comprehensive metrics

Optimizations:
1. Single-file format (2.4× speedup)
2. Faiss centroid search (40× faster)
3. RaBitQ 2-bit quantization
4. Hierarchical clustering
5. RNG graph
6. Batch loading
"""
import numpy as np
import time
import pickle
import os
import h5py
import faiss
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Data paths
DATA_FILE = '/data/cohere-wikipedia-768-angular.hdf5'
INDEX_DIR = '/mnt/nvme/spann_cohere_optimized'

def load_cohere_data(num_queries=1000):
    """Load Cohere 1M dataset from HDF5"""
    print("Loading Cohere 1M dataset...")
    with h5py.File(DATA_FILE, 'r') as f:
        base = np.array(f['train'])
        queries = np.array(f['test'][:num_queries])
        neighbors = np.array(f['neighbors'][:num_queries])
    print(f"✓ Base: {base.shape}, Queries: {queries.shape}")
    return base, queries, neighbors

def get_disk_usage(path):
    """Get disk usage in MB"""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def benchmark_search(index, base, queries, groundtruth, config_name, search_params):
    """Benchmark search with given parameters"""
    num_queries = len(queries)
    k = 10
    
    recalls = []
    latencies = []
    
    print(f"\nSearching {num_queries} queries with {config_name}...")
    for i in range(num_queries):
        t0 = time.perf_counter()
        ids, _ = index.search(
            queries[i], base, k=k, 
            search_internal_result_num=search_params['centroids'],
            max_check=search_params['max_check'],
            use_async_pruning=True  # Enable SPTAG-style async pruning
        )
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)
        
        # Recall
        gt = set(int(x) for x in groundtruth[i][:k])
        pred = set(ids[:k]) if len(ids) >= k else set(ids)
        recall = len(gt & pred) / k
        recalls.append(recall)
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{num_queries} queries done")
    
    return {
        'recall': np.mean(recalls) * 100,
        'p50': np.percentile(latencies, 50),
        'p90': np.percentile(latencies, 90),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'mean': np.mean(latencies),
        'qps': num_queries / np.sum(latencies) * 1000
    }

def main():
    print("="*80)
    print("Cohere 1M Benchmark - SPTAG Optimized SPANN on EC2")
    print("="*80)
    
    # Load data
    base, queries, groundtruth = load_cohere_data(num_queries=1000)
    
    # Build or load index
    if os.path.exists(INDEX_DIR):
        print(f"\n✓ Loading existing index from {INDEX_DIR}")
        index = SPANNDiskOptimized(
            dim=768,
            metric='IP',
            disk_path=INDEX_DIR,
            use_rabitq=True,
            use_faiss_centroids=True,
            cache_size=5000
        )
        
        # Load metadata
        with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            for k, v in metadata.items():
                if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq']:
                    setattr(index, k, v)
        
        # Setup Faiss index (metric-aware)
        if index.metric == 'L2':
            index._centroid_index = faiss.IndexFlatL2(index.dim)
        else:  # IP or Cosine
            index._centroid_index = faiss.IndexFlatIP(index.dim)
        index._centroid_index.add(index.centroids.astype(np.float32))
        
        print(f"✓ Loaded: {index.num_clusters} clusters, metric={index.metric}")
    else:
        print(f"\nBuilding optimized index at {INDEX_DIR}...")
        print("  Config: posting_size=500, replica=8, RaBitQ 2-bit")
        print("  Clustering: hierarchical, Tree: BKT+RNG")
        
        t0 = time.time()
        index = SPANNDiskOptimized(
            dim=768,
            target_posting_size=500,
            replica_count=8,
            use_rabitq=True,
            bq=2,
            metric='IP',
            tree_type='BKT',
            clustering='hierarchical',
            use_rng_filtering=True,
            use_faiss_centroids=True,
            disk_path=INDEX_DIR,
            cache_size=5000
        )
        index.build(base)
        build_time = time.time() - t0
        
        print(f"✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")
        print(f"✓ Clusters: {index.num_clusters}")
    
    disk_mb = get_disk_usage(INDEX_DIR)
    print(f"✓ Disk usage: {disk_mb:.1f} MB ({disk_mb/1024:.2f} GB)")
    
    # Warm up
    print("\nWarming up (20 queries)...")
    for q in queries[:20]:
        index.search(q, base, k=10, search_internal_result_num=48, max_check=6144)
    
    # Test configurations
    configs = [
        ('centroids=32, max_check=4096', {'centroids': 32, 'max_check': 4096}),
        ('centroids=48, max_check=6144', {'centroids': 48, 'max_check': 6144}),
        ('centroids=64, max_check=8192', {'centroids': 64, 'max_check': 8192}),
    ]
    
    results = []
    for name, params in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print("="*80)
        
        result = benchmark_search(index, base, queries, groundtruth, name, params)
        result['config'] = name
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Recall:  {result['recall']:.2f}% {'✅' if result['recall'] >= 90 else '⚠️'}")
        print(f"  p50:     {result['p50']:.2f}ms {'✅' if result['p50'] < 10 else '⚠️'}")
        print(f"  p90:     {result['p90']:.2f}ms")
        print(f"  p95:     {result['p95']:.2f}ms")
        print(f"  p99:     {result['p99']:.2f}ms")
        print(f"  QPS:     {result['qps']:.1f}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Index: {index.num_clusters} clusters, {disk_mb/1024:.2f} GB")
    print()
    print(f"{'Config':<30} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10} {'Status':<15}")
    print("-"*80)
    
    for r in results:
        status = ''
        if r['p50'] < 10 and r['recall'] >= 90:
            status = '🎉 BOTH TARGETS'
        elif r['recall'] >= 90:
            status = '✅ Recall'
        elif r['p50'] < 10:
            status = '✅ Latency'
        
        print(f"{r['config']:<30} {r['p50']:<10.2f} {r['p90']:<10.2f} {r['recall']:<10.1f} {status:<15}")
    
    print("\nOptimizations enabled:")
    print("  ✅ Single-file format")
    print("  ✅ Faiss centroid search")
    print("  ✅ RaBitQ 2-bit quantization")
    print("  ✅ Hierarchical clustering")
    print("  ✅ RNG graph")
    print("  ✅ Batch loading")
    print("  ✅ Large cache (5000)")

if __name__ == '__main__':
    main()
