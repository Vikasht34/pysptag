#!/usr/bin/env python3
"""Simple latency profiling by adding timing to search."""

import numpy as np
import h5py
import sys
import os
import pickle
sys.path.insert(0, '.')

# Add timing globals
timing_data = {
    'centroid': [],
    'distance': [],
    'rerank': []
}

def profile_config(config_name, index_dir):
    """Profile a configuration."""
    global timing_data
    timing_data = {'centroid': [], 'distance': [], 'rerank': []}
    
    print(f"\n{'='*70}")
    print(f"{config_name}")
    print('='*70)
    
    if not os.path.exists(index_dir):
        print("  Index not found")
        return None
    
    # Load data
    with h5py.File('data/cohere/documents-1m.hdf5', 'r') as f:
        base = f['train'][:]
        queries = f['test'][:100]
        groundtruth = f['neighbors'][:100]
    
    # Load index
    from src.index.spann_disk_optimized import SPANNDiskOptimized
    
    use_rabitq = False
    bq = None
    if '1bit' in config_name:
        use_rabitq, bq = True, 1
    elif '2bit' in config_name:
        use_rabitq, bq = True, 2
    elif '4bit' in config_name:
        use_rabitq, bq = True, 4
    
    index = SPANNDiskOptimized(
        dim=768, use_rabitq=use_rabitq, bq=bq, metric='IP',
        disk_path=index_dir, cache_size=2000
    )
    
    with open(os.path.join(index_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        for k, v in metadata.items():
            if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq']:
                setattr(index, k, v)
    
    # Run queries
    recalls = []
    latencies = []
    
    import time
    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        result_indices, _ = index.search(query, base, k=10, search_internal_result_num=96, max_check=12288)
        latencies.append((time.perf_counter() - t0) * 1000)
        
        gt = set(groundtruth[i][:10])
        found = set(result_indices[:10])
        recalls.append(len(gt & found) / 10)
    
    print(f"\nRecall@10: {np.mean(recalls)*100:.1f}%")
    print(f"Latency p50: {np.percentile(latencies, 50):.2f}ms")
    
    if timing_data['centroid']:
        print(f"\nTiming Breakdown (p50):")
        print(f"  Centroid search:  {np.percentile(timing_data['centroid'], 50):.2f}ms")
        print(f"  Distance compute: {np.percentile(timing_data['distance'], 50):.2f}ms")
        print(f"  Reranking:        {np.percentile(timing_data['rerank'], 50):.2f}ms")
    
    return {
        'recall': np.mean(recalls) * 100,
        'latency': np.percentile(latencies, 50),
        'centroid': np.percentile(timing_data['centroid'], 50) if timing_data['centroid'] else 0,
        'distance': np.percentile(timing_data['distance'], 50) if timing_data['distance'] else 0,
        'rerank': np.percentile(timing_data['rerank'], 50) if timing_data['rerank'] else 0
    }

if __name__ == '__main__':
    configs = [
        ('No-Quant', '/tmp/cohere_e2e_NoQuant'),
        ('1-bit', '/tmp/cohere_e2e_1bit'),
        ('2-bit', '/tmp/cohere_e2e_2bit'),
        ('4-bit', '/tmp/cohere_e2e_4bit'),
    ]
    
    results = {}
    for name, path in configs:
        results[name] = profile_config(name, path)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"{'Config':<12} {'Recall':<10} {'Latency':<10}")
    print('-'*40)
    for name, r in results.items():
        if r:
            print(f"{name:<12} {r['recall']:>6.1f}%    {r['latency']:>6.2f}ms")
