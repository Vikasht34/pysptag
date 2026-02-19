"""
Build and test HNSW on GloVe dataset
"""
import sys
import numpy as np
import h5py
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("Loading GloVe data...")
with h5py.File('/Users/viktari/pysptag/data/glove/glove-200-angular.hdf5', 'r') as f:
    base = f['train'][:]
    queries = f['test'][:]
    groundtruth = f['neighbors'][:]
print(f"Loaded: base={base.shape}, queries={queries.shape}, GT={groundtruth.shape}")

# Normalize for cosine similarity
print("Normalizing vectors...")
base = base / np.linalg.norm(base, axis=1, keepdims=True)
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

print("\nBuilding index with HNSW...")
index = SPANNDiskOptimized(
    dim=200,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='Cosine',
    use_hnsw_centroids=True,
    hnsw_m=16,
    hnsw_ef_construction=200,
    clustering='hierarchical',
    use_rng_filtering=True,
    use_faiss_centroids=False,
    disk_path='/tmp/glove_index',
    cache_size=2000
)

index.build(base)
print(f"✓ Index built: {index.num_clusters} clusters")

# Test parameters
num_queries = 1000
centroid_counts = [32, 64, 128, 256, 512]
max_checks = [2048, 4096, 8192, 16384, 32768, 49152, 1000000]

print(f"\nTesting with {num_queries} queries")
print(f"{'Centroids':>10} {'MaxCheck':>10} {'Recall@100':>12} {'Avg(ms)':>10}")
print("-" * 50)

sample_queries = queries[:num_queries]
sample_gt = groundtruth[:num_queries]

import time

for num_centroids in centroid_counts:
    for max_check in max_checks:
        recalls = []
        latencies = []
        
        for i, query in enumerate(sample_queries):
            t0 = time.perf_counter()
            indices, dists = index.search(
                query, base, k=100,
                search_internal_result_num=num_centroids,
                max_check=max_check
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            
            gt = set(sample_gt[i][:100])
            found = set(indices[:100])
            recalls.append(len(gt & found) / 100)
        
        avg_recall = np.mean(recalls) * 100
        avg_latency = np.mean(latencies)
        
        print(f"{num_centroids:>10} {max_check:>10} {avg_recall:>11.2f}% {avg_latency:>10.2f}")

print("\nDone!")
