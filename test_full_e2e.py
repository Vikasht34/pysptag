#!/usr/bin/env python3
"""End-to-end correctness test for all quantization levels on SIFT and Cohere."""

import numpy as np
import h5py
import sys
import os
import time
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

def test_dataset(name, data_file, dim, num_queries=100):
    """Test all quantization levels on a dataset."""
    print(f"\n{'='*70}")
    print(f"Testing {name} dataset (dim={dim})")
    print('='*70)
    
    # Load data
    with h5py.File(data_file, 'r') as f:
        base = f['train'][:]
        queries = f['test'][:num_queries]
        groundtruth = f['neighbors'][:num_queries]
    
    results = []
    
    # Test configurations: (bq, use_rabitq, name)
    configs = [
        (None, False, 'No-Quant'),
        (1, True, '1-bit'),
        (2, True, '2-bit'),
        (4, True, '4-bit'),
    ]
    
    for bq, use_rabitq, config_name in configs:
        index_dir = f'/tmp/{name.lower()}_e2e_{config_name.replace("-", "")}'
        
        print(f"\n{config_name}:")
        print(f"  Building index...")
        
        # Build index
        t0 = time.time()
        index = SPANNDiskOptimized(
            dim=dim,
            target_posting_size=500,
            replica_count=8,
            use_rabitq=use_rabitq,
            bq=bq,
            metric='IP',
            tree_type='BKT',
            clustering='hierarchical',
            use_rng_filtering=True,
            use_faiss_centroids=False,
            disk_path=index_dir,
            cache_size=2000
        )
        index.build(base)
        build_time = time.time() - t0
        
        # Get disk size
        total_size = 0
        for root, dirs, files in os.walk(index_dir):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        disk_mb = total_size / (1024 * 1024)
        
        # Search with moderate params
        num_centroids = 96
        max_check = 12288
        
        recalls = []
        latencies = []
        
        for i, query in enumerate(queries):
            t0 = time.perf_counter()
            indices, dists = index.search(
                query, base, k=10,
                search_internal_result_num=num_centroids,
                max_check=max_check
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            
            gt = set(groundtruth[i][:10])
            found = set(indices[:10])
            recalls.append(len(gt & found) / 10)
        
        recall = np.mean(recalls) * 100
        p50 = np.percentile(latencies, 50)
        
        print(f"  Build: {build_time:.1f}s")
        print(f"  Disk: {disk_mb:.1f}MB")
        print(f"  Recall@10: {recall:.1f}%")
        print(f"  Latency p50: {p50:.1f}ms")
        
        results.append({
            'config': config_name,
            'recall': recall,
            'latency': p50,
            'disk_mb': disk_mb,
            'build_time': build_time
        })
    
    # Summary table
    print(f"\n{name} Summary:")
    print(f"{'Config':<12} {'Recall@10':<12} {'p50 (ms)':<12} {'Disk (MB)':<12}")
    print('-' * 50)
    for r in results:
        print(f"{r['config']:<12} {r['recall']:>6.1f}%     {r['latency']:>6.1f}      {r['disk_mb']:>7.1f}")
    
    return results

if __name__ == '__main__':
    # Test Cohere
    cohere_results = test_dataset(
        'Cohere',
        'data/cohere/documents-1m.hdf5',
        dim=768,
        num_queries=100
    )
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - Cohere 1M Dataset")
    print('='*70)
    for r in cohere_results:
        print(f"  {r['config']}: {r['recall']:.1f}% recall, {r['latency']:.1f}ms p50, {r['disk_mb']:.0f}MB")
