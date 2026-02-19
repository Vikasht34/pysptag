"""
Test Cohere 1M without quantization
"""
import numpy as np
import h5py
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_FILE = 'data/cohere/documents-1m.hdf5'
INDEX_DIR = '/tmp/cohere_noquant'
NUM_QUERIES = 100

print("="*80)
print("Cohere 1M - No Quantization Test")
print("="*80)

print("\nLoading Cohere 1M...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

print("\nBuilding index (no quantization)...")
t0 = time.time()
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='IP',
    tree_type='BKT',
    clustering='hierarchical',
    use_rng_filtering=True,
    use_faiss_centroids=False,
    disk_path=INDEX_DIR,
    cache_size=2000
)
index.build(base)
build_time = time.time() - t0
print(f"✓ Built in {build_time:.1f}s")

print("\nWarming up...")
for q in queries[:20]:
    index.search(q, base, k=10, search_internal_result_num=48, max_check=6144)

print(f"\nTesting ({NUM_QUERIES} queries)...")
recalls = []
latencies = []

for i, query in enumerate(queries):
    t0 = time.perf_counter()
    indices, dists = index.search(
        query, base, k=10, 
        search_internal_result_num=48,
        max_check=6144
    )
    latencies.append((time.perf_counter() - t0) * 1000)
    
    gt = set(groundtruth[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{NUM_QUERIES}...")

avg_recall = np.mean(recalls) * 100
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
avg = np.mean(latencies)

print(f"\n{'='*80}")
print("Results (No Quantization):")
print(f"  Recall:  {avg_recall:.2f}%")
print(f"  p50:     {p50:.2f}ms")
print(f"  p90:     {p90:.2f}ms")
print(f"  avg:     {avg:.2f}ms")
print("="*80)
