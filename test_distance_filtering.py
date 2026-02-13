"""Test distance-based centroid filtering optimization"""
import numpy as np
import time
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Generate synthetic data
print("Generating synthetic data...")
np.random.seed(42)
n_base = 10000
n_queries = 100
dim = 128

base = np.random.randn(n_base, dim).astype(np.float32)
base /= np.linalg.norm(base, axis=1, keepdims=True)
queries = np.random.randn(n_queries, dim).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

print(f"Base: {base.shape}, Queries: {queries.shape}")

# Test configurations
configs = [
    ("Baseline (no filtering)", 10000.0),  # Very high = no filtering
    ("Aggressive (100)", 100.0),
    ("Very aggressive (10)", 10.0),
]

for name, max_dist_ratio in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Build index
    index = SPANNDiskOptimized(
        dim=128,
        target_posting_size=500,
        replica_count=8,
        use_rabitq=False,
        clustering='kmeans'
    )
    
    print("Building index...")
    index.build(base)
    print(f"Built {len(index.centroids)} clusters")
    
    # Search
    latencies = []
    
    for i, query in enumerate(queries):
        start = time.perf_counter()
        dists, indices = index.search(
            query, base, k=10,
            search_internal_result_num=64,
            max_check=4096,
            max_dist_ratio=max_dist_ratio
        )
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
    
    # Results
    latencies = np.array(latencies)
    
    print(f"\nResults:")
    print(f"  p50 latency: {np.percentile(latencies, 50):.2f}ms")
    print(f"  p90 latency: {np.percentile(latencies, 90):.2f}ms")
    print(f"  p99 latency: {np.percentile(latencies, 99):.2f}ms")
    print(f"  Cache hits:  {index._cache_hits}")
    print(f"  Cache miss:  {index._cache_misses}")
