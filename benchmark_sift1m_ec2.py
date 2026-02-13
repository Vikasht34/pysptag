#!/usr/bin/env python3
"""
SIFT 1M Benchmark on EC2 - Comprehensive metrics with quantization comparison
Tests: No quant, 1-bit, 2-bit, 4-bit RaBitQ
"""
import numpy as np
import time
import pickle
import os
from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Data paths
DATA_DIR = '/data/sift'
BASE_FILE = f'{DATA_DIR}/sift_base.fvecs'
QUERY_FILE = f'{DATA_DIR}/sift_query.fvecs'
GT_FILE = f'{DATA_DIR}/sift_groundtruth.ivecs'

def load_ivecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            d = np.fromfile(f, dtype=np.int32, count=1)
            if len(d) == 0:
                break
            vec = np.fromfile(f, dtype=np.int32, count=d[0])
            data.append(vec)
    return np.array(data)

def get_disk_usage(path):
    """Get disk usage in MB"""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def benchmark_search(index, base, queries, groundtruth, num_queries=1000, k=10, max_check=4096):
    """Search benchmark with detailed latency breakdown"""
    recalls = []
    latencies = []
    bytes_read_list = []
    
    # Detailed timing
    centroid_search_times = []
    posting_load_times = []
    distance_comp_times = []
    sort_times = []
    
    print(f"\nSearching {num_queries} queries...")
    for i in range(num_queries):
        # Reset counter
        index._bytes_read = 0
        
        query = queries[i]
        t_total = time.perf_counter()
        
        # Stage 1: Find nearest centroids
        t0 = time.perf_counter()
        if hasattr(index, 'rng') and len(index.rng.graph) > 0:
            if index.tree_type == 'BKT':
                from src.core.bktree_rng_search import bktree_rng_search
                nearest_centroids = bktree_rng_search(
                    query, index.centroids, index.tree.tree_roots,
                    index.tree.tree_start, index.rng.graph,
                    64, index.metric
                )
            else:
                nearest_centroids = index.tree.search(query, index.centroids, 64, index.metric)
        else:
            if index.metric == 'L2':
                centroid_dists = np.sum((index.centroids - query) ** 2, axis=1)
            else:
                centroid_dists = -np.dot(index.centroids, query)
            sorted_idx = np.argsort(centroid_dists)
            nearest_centroids = sorted_idx[:64]
        centroid_search_times.append((time.perf_counter() - t0) * 1000)
        
        # Stage 2: Load posting lists
        t0 = time.perf_counter()
        candidates = []
        for cid in nearest_centroids[:max_check]:
            posting = index._load_posting_mmap(cid)
            if posting is not None and posting[0] is not None:
                candidates.extend(posting[0])
        posting_load_times.append((time.perf_counter() - t0) * 1000)
        
        # Stage 3: Distance computation
        t0 = time.perf_counter()
        candidates = list(set(candidates))
        if len(candidates) > 0:
            candidate_vecs = base[candidates]
            dists = np.sum((candidate_vecs - query) ** 2, axis=1)
        else:
            dists = np.array([])
        distance_comp_times.append((time.perf_counter() - t0) * 1000)
        
        # Stage 4: Sort and get top-k
        t0 = time.perf_counter()
        if len(dists) >= k:
            top_k_idx = np.argpartition(dists, k)[:k]
            top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
            ids = [candidates[idx] for idx in top_k_idx]
        else:
            ids = candidates
        sort_times.append((time.perf_counter() - t0) * 1000)
        
        latency = (time.perf_counter() - t_total) * 1000
        latencies.append(latency)
        bytes_read_list.append(index._bytes_read)
        
        # Recall
        recall = len(set(ids) & set(int(x) for x in groundtruth[i][:k])) / k
        recalls.append(recall)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_queries} queries done")
    
    return {
        'recall': np.mean(recalls) * 100,
        'latency': {
            'p50': np.percentile(latencies, 50),
            'p90': np.percentile(latencies, 90),
            'p99': np.percentile(latencies, 99),
            'mean': np.mean(latencies)
        },
        'breakdown': {
            'centroid_search': np.mean(centroid_search_times),
            'posting_load': np.mean(posting_load_times),
            'distance_comp': np.mean(distance_comp_times),
            'sort': np.mean(sort_times)
        },
        'bytes_per_query_kb': np.mean(bytes_read_list) / 1024,
        'postings_per_query': np.mean([len(set(range(min(max_check, 64)))) for _ in range(num_queries)])
    }
    """Detailed search benchmark with latency breakdown"""
    recalls = []
    
    # Timing breakdowns
    fetch_times = []
    distance_times = []
    rerank_times = []
    total_times = []
    
    # Disk I/O metrics
    bytes_read_per_query = []
    postings_accessed = []
    
    print(f"\nSearching {num_queries} queries...")
    for i in range(num_queries):
        # Instrument search
        t_start = time.perf_counter()
        
        # Search with instrumentation
        query = queries[i]
        
        # Stage 1: Fetch posting lists
        t0 = time.perf_counter()
        # Get top clusters from centroids
        dists = np.sum((query - index.centroids) ** 2, axis=1)
        cluster_ids = np.argpartition(dists, min(max_check, len(dists)-1))[:max_check]
        
        # Load postings
        candidates = []
        bytes_read = 0
        for cid in cluster_ids:
            posting = index._load_posting_mmap(cid)
            candidates.extend(posting)
            bytes_read += len(posting) * 4  # 4 bytes per int32
        
        t1 = time.perf_counter()
        fetch_time = (t1 - t0) * 1000
        
        # Stage 2: Distance computation
        t0 = time.perf_counter()
        candidates = list(set(candidates))[:max_check * 10]  # Dedupe
        if len(candidates) > 0:
            candidate_vecs = base[candidates]
            dists = np.sum((candidate_vecs - query) ** 2, axis=1)
        else:
            dists = np.array([])
        t1 = time.perf_counter()
        distance_time = (t1 - t0) * 1000
        
        # Stage 3: Reranking (only for quantized)
        rerank_time = 0.0
        if index.use_rabitq and hasattr(index, 'rabitq') and index.rabitq is not None:
            t0 = time.perf_counter()
            # Rerank top candidates with full precision
            if len(dists) >= k * 2:
                top_rerank = min(k * 2, len(dists))
                rerank_idx = np.argpartition(dists, top_rerank)[:top_rerank]
                rerank_vecs = candidate_vecs[rerank_idx]
                rerank_dists = np.sum((rerank_vecs - query) ** 2, axis=1)
                top_k_idx = np.argpartition(rerank_dists, k)[:k]
                top_k_idx = top_k_idx[np.argsort(rerank_dists[top_k_idx])]
                result_ids = [candidates[rerank_idx[idx]] for idx in top_k_idx]
            else:
                result_ids = candidates
            t1 = time.perf_counter()
            rerank_time = (t1 - t0) * 1000
        else:
            # No reranking for full precision, just sort
            if len(dists) >= k:
                top_k_idx = np.argpartition(dists, k)[:k]
                top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
                result_ids = [candidates[idx] for idx in top_k_idx]
            else:
                result_ids = candidates
        
        t_end = time.perf_counter()
        total_time = (t_end - t_start) * 1000
        
        # Metrics
        fetch_times.append(fetch_time)
        distance_times.append(distance_time)
        rerank_times.append(rerank_time)
        total_times.append(total_time)
        bytes_read_per_query.append(bytes_read)
        postings_accessed.append(len(cluster_ids))
        
        # Recall
        recall = len(set(result_ids) & set(int(x) for x in groundtruth[i][:k])) / k
        recalls.append(recall)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_queries} queries done")
    
    return {
        'recall': np.mean(recalls) * 100,
        'total_latency': {
            'p50': np.percentile(total_times, 50),
            'p90': np.percentile(total_times, 90),
            'p99': np.percentile(total_times, 99),
            'mean': np.mean(total_times)
        },
        'fetch_latency': {
            'p50': np.percentile(fetch_times, 50),
            'p99': np.percentile(fetch_times, 99),
            'mean': np.mean(fetch_times)
        },
        'distance_latency': {
            'p50': np.percentile(distance_times, 50),
            'p99': np.percentile(distance_times, 99),
            'mean': np.mean(distance_times)
        },
        'rerank_latency': {
            'p50': np.percentile(rerank_times, 50),
            'p99': np.percentile(rerank_times, 99),
            'mean': np.mean(rerank_times)
        },
        'disk_io': {
            'bytes_per_query_kb': np.mean(bytes_read_per_query) / 1024,
            'postings_accessed': np.mean(postings_accessed)
        }
    }

def run_benchmark(quant_bits, preload=False):
    """Run full benchmark for given quantization"""
    print(f"\n{'='*70}")
    print(f"Benchmark: {quant_bits}-bit quantization" if quant_bits > 0 else "Benchmark: No quantization")
    print(f"Preload postings: {preload}")
    print(f"{'='*70}")
    
    disk_path = f'/tmp/sift1m_{quant_bits}bit' if quant_bits > 0 else '/tmp/sift1m_noq'
    
    # Load data
    print("\n[1/3] Loading data...")
    base = load_fvecs(BASE_FILE)
    queries = load_fvecs(QUERY_FILE)
    groundtruth = load_ivecs(GT_FILE)
    print(f"  Base: {base.shape}, Queries: {queries.shape}")
    
    # Build or load index
    metadata_path = f'{disk_path}/metadata.pkl'
    if not os.path.exists(metadata_path):
        print(f"\n[2/3] Building index...")
        t0 = time.time()
        index = SPANNDiskOptimized(
            dim=128,
            metric='L2',
            tree_type='BKT',
            replica_count=8,
            use_rng_filtering=True,
            clustering='hierarchical',
            target_posting_size=800,
            disk_path=disk_path,
            use_rabitq=(quant_bits > 0),
            bq=quant_bits if quant_bits > 0 else 4,
            preload_postings=preload
        )
        index.build(base)
        build_time = time.time() - t0
        print(f"  Build time: {build_time:.1f}s")
    else:
        print(f"\n[2/3] Loading existing index...")
        t0 = time.time()
        index = SPANNDiskOptimized(
            dim=128,
            metric='L2',
            tree_type='BKT',
            replica_count=8,
            use_rng_filtering=True,
            clustering='hierarchical',
            target_posting_size=800,
            disk_path=disk_path,
            use_rabitq=(quant_bits > 0),
            bq=quant_bits if quant_bits > 0 else 4,
            preload_postings=preload
        )
        with open(f'{disk_path}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            index.centroids = metadata['centroids']
            index.tree = metadata['tree']
            index.rng = metadata['rng']
            index.num_clusters = metadata['num_clusters']
        
        if preload:
            index.preload_all_postings()
        
        load_time = time.time() - t0
        build_time = None
        print(f"  Load time: {load_time:.1f}s")
    
    # Disk usage
    disk_mb = get_disk_usage(disk_path)
    
    # Benchmark
    print(f"\n[3/3] Running benchmark...")
    results = benchmark_search(index, base, queries, groundtruth, num_queries=1000)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {quant_bits}-bit" if quant_bits > 0 else "RESULTS: No quantization")
    print(f"{'='*70}")
    if build_time:
        print(f"Build time:        {build_time:.1f}s")
    print(f"Disk usage:        {disk_mb:.1f} MB")
    print(f"Clusters:          {index.num_clusters}")
    print(f"\nRecall@10:         {results['recall']:.2f}%")
    print(f"\nLatency:")
    print(f"  p50:             {results['latency']['p50']:.2f} ms")
    print(f"  p90:             {results['latency']['p90']:.2f} ms")
    print(f"  p99:             {results['latency']['p99']:.2f} ms")
    print(f"  mean:            {results['latency']['mean']:.2f} ms")
    print(f"\nLatency Breakdown (mean):")
    print(f"  Centroid search: {results['breakdown']['centroid_search']:.2f} ms")
    print(f"  Posting load:    {results['breakdown']['posting_load']:.2f} ms")
    print(f"  Distance comp:   {results['breakdown']['distance_comp']:.2f} ms")
    print(f"  Sort:            {results['breakdown']['sort']:.2f} ms")
    print(f"\nDisk I/O per query:")
    print(f"  Data read:       {results['bytes_per_query_kb']:.1f} KB")
    print(f"{'='*70}\n")
    
    return results

if __name__ == '__main__':
    print("SIFT 1M Benchmark - EC2")
    print("Testing: No quant, 1-bit, 2-bit, 4-bit")
    
    configs = [
        (0, False),   # No quantization, no preload
        (1, False),   # 1-bit
        (2, False),   # 2-bit
        (4, False),   # 4-bit
    ]
    
    all_results = {}
    for quant_bits, preload in configs:
        key = f"{quant_bits}bit" if quant_bits > 0 else "noq"
        all_results[key] = run_benchmark(quant_bits, preload)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Config':<12} {'Recall':<10} {'p50 (ms)':<12} {'p99 (ms)':<12} {'Disk I/O (KB)':<15}")
    print("-"*70)
    for key, res in all_results.items():
        print(f"{key:<12} {res['recall']:>6.2f}%   {res['latency']['p50']:>8.2f}     "
              f"{res['latency']['p99']:>8.2f}     {res['bytes_per_query_kb']:>10.1f}")
    print("="*70)
