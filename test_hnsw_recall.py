"""
Test HNSW centroid search - load existing index and measure recall
"""
import sys
import numpy as np
import time
import pickle
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized
import hnswlib

print("Loading Cohere data...")
base = np.fromfile('/Users/viktari/cohere_data/base.bin', dtype=np.float32)[2:].reshape(-1, 768)
queries = np.fromfile('/Users/viktari/cohere_data/query.bin', dtype=np.float32)[2:].reshape(-1, 768)
gt_raw = np.fromfile('/Users/viktari/cohere_data/groundtruth.bin', dtype=np.int32)[2:]
groundtruth = gt_raw[:len(gt_raw) // 100 * 100].reshape(-1, 100)
print(f"Loaded: base={base.shape}, queries={queries.shape}, GT={groundtruth.shape}")

print("\nLoading index...")
index = SPANNDiskOptimized(dim=768, disk_path='/tmp/cohere_index_new', cache_size=2000)

with open('/tmp/cohere_index_new/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    for k, v in metadata.items():
        if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq', '_hnsw_index']:
            setattr(index, k, v)

# Load HNSW
index._hnsw_index = hnswlib.Index(space='ip', dim=768)
index._hnsw_index.load_index('/tmp/cohere_index_new/hnsw_centroids.bin', max_elements=index.num_clusters)
print(f"Index loaded: {index.num_clusters} clusters, HNSW enabled")

# Debug: Test a single search
print("\nDebug: Testing single search...")
query = queries[0]
indices, dists = index.search(query, base, k=10, search_internal_result_num=32, max_check=4096)
print(f"Returned indices: {indices[:10]}")
print(f"Returned dists: {dists[:10]}")
print(f"GT for query 0: {groundtruth[0][:10]}")
print(f"Overlap: {len(set(indices[:10]) & set(groundtruth[0][:10]))}/10")

# Test parameters
num_queries = 1000
centroid_counts = [32, 64, 128, 256, 512]
max_checks = [2048, 4096, 8192, 16384, 32768, 49152, 1000000]

print(f"\nTesting with {num_queries} queries")
print(f"{'Centroids':>10} {'MaxCheck':>10} {'Recall@100':>12} {'Avg(ms)':>10}")
print("-" * 50)

sample_queries = queries[:num_queries]
sample_gt = groundtruth[:num_queries]

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
