"""Test different search parameters for higher recall"""
import numpy as np
import h5py
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_FILE = 'data/cohere/documents-1m.hdf5'
INDEX_DIR = '/tmp/cohere_1bit'
NUM_QUERIES = 100

print("Loading data...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print("Loading index...")
index = SPANNDiskOptimized(
    dim=768,
    use_rabitq=True,
    bq=1,
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

# Test different parameter combinations
configs = [
    # (search_internal_result_num, max_check, name)
    (48, 6144, 'baseline'),
    (64, 8192, 'more_centroids'),
    (96, 12288, 'even_more'),
    (128, 16384, 'aggressive'),
]

for num_centroids, max_check, name in configs:
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  Centroids: {num_centroids}, Max check: {max_check}")
    print("="*60)
    
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
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    
    print(f"Recall: {avg_recall:.2f}%")
    print(f"p50: {p50:.2f}ms, p90: {p90:.2f}ms")
