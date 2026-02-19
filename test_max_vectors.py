"""
Test Cohere with each centroid count using ALL vectors in postings (no max_check limit)
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_DIR = '/Users/viktari/cohere_data'
INDEX_DIR = '/tmp/cohere_noquant'
NUM_QUERIES = 10000
SAMPLE_SIZE = 1000

print("="*80)
print("Cohere 1M - Max Vectors Test - Recall@100")
print("="*80)

print("\nLoading data...")
base = np.fromfile(f'{DATA_DIR}/cohere_base.bin', dtype=np.float32)[:768000000].reshape(-1, 768)
queries = np.fromfile(f'{DATA_DIR}/cohere_query.bin', dtype=np.float32)[:768*NUM_QUERIES].reshape(-1, 768)
groundtruth = np.fromfile(f'{DATA_DIR}/cohere_groundtruth.bin', dtype=np.int32)[2:2+100*NUM_QUERIES].reshape(-1, 100)

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

print("\nLoading index...")
index = SPANNDiskOptimized(
    dim=768,
    use_rabitq=False,
    metric='IP',
    disk_path=INDEX_DIR,
    cache_size=2000
)

import pickle
with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)
    for k, v in metadata.items():
        if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq']:
            setattr(index, k, v)

print(f"✓ Loaded: {index.num_clusters} clusters")

# Sample queries
np.random.seed(42)
sample_indices = np.random.choice(NUM_QUERIES, SAMPLE_SIZE, replace=False)
sample_queries = queries[sample_indices]
sample_gt = groundtruth[sample_indices]

print(f"Using {SAMPLE_SIZE} sampled queries from {NUM_QUERIES} total")

# Test with different centroid counts, using ALL vectors (max_check = 1M)
centroid_counts = [32, 64, 128, 256, 512, 1024, 2048]

print("\n" + "="*80)
print("Max Vectors Test Results")
print("="*80)
print(f"{'Centroids':>10} {'MaxCheck':>10} {'Recall@100':>12} {'P50(ms)':>10} {'P90(ms)':>10} {'Avg(ms)':>10}")
print("-"*80)
sys.stdout.flush()

for num_centroids in centroid_counts:
    max_check = 1000000  # No limit - search all vectors
    
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
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    avg = np.mean(latencies)
    
    result = f"{num_centroids:>10} {max_check:>10} {avg_recall:>11.2f}% {p50:>10.2f} {p90:>10.2f} {avg:>10.2f}"
    print(result)
    sys.stdout.flush()

print("="*80)
