"""
Test Cohere 1M locally on Mac
"""
import numpy as np
import h5py
import time
import os
import faiss
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_FILE = 'data/cohere/documents-1m.hdf5'
INDEX_DIR = '/tmp/cohere_local'

print("="*80)
print("Cohere 1M Local Test (Mac)")
print("="*80)

# Load data
print("\nLoading Cohere 1M...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:1000]
    ground_truth = f['neighbors'][:1000]

print(f"  Base: {base.shape}")
print(f"  Queries: {queries.shape}")
print(f"  Ground truth: {ground_truth.shape}")

# Check if index exists
if os.path.exists(INDEX_DIR):
    print(f"\n⚠️  Index exists at {INDEX_DIR}")
    print("   Loading existing index...")
    
    index = SPANNDiskOptimized(
        dim=768,
        metric='IP',
        disk_path=INDEX_DIR,
        use_rabitq=True,
        use_faiss_centroids=True,
        cache_size=2000
    )
    
    import pickle
    with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
        for k, v in metadata.items():
            if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq']:
                setattr(index, k, v)
    
    # Setup Faiss with correct metric
    if index.metric == 'L2':
        index._centroid_index = faiss.IndexFlatL2(index.dim)
    else:
        index._centroid_index = faiss.IndexFlatIP(index.dim)
    index._centroid_index.add(index.centroids.astype(np.float32))
    
    print(f"  ✓ Loaded: {index.num_clusters} clusters, metric={index.metric}")
else:
    print(f"\nBuilding new index at {INDEX_DIR}...")
    print("  Config: posting_size=500, replica=8, RaBitQ 2-bit")
    
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
        cache_size=2000
    )
    
    t0 = time.time()
    index.build(base)
    print(f"  ✓ Built in {time.time()-t0:.1f}s")

# Warm up
print("\nWarming up (20 queries)...")
for q in queries[:20]:
    index.search(q, base, k=10, search_internal_result_num=48, max_check=6144, use_async_pruning=True)

# Test different configurations
configs = [
    ('centroids=32', 32, 6144),
    ('centroids=48', 48, 6144),
    ('centroids=64', 64, 8192),
]

for name, num_centroids, max_check in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {name}, max_check={max_check}")
    print("="*80)
    
    latencies = []
    recalls = []
    
    for i in range(1000):
        t0 = time.perf_counter()
        ids, _ = index.search(
            queries[i], base, k=10,
            search_internal_result_num=num_centroids,
            max_check=max_check,
            use_async_pruning=True,
            max_vectors_per_posting=500
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        
        gt = set(ground_truth[i, :10])
        pred = set(ids[:10])
        recalls.append(len(gt & pred) / 10.0)
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/1000...")
    
    mean_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    avg = np.mean(latencies)
    
    print(f"\nResults:")
    print(f"  Recall:  {mean_recall:.2f}% {'✅' if mean_recall >= 90 else '⚠️'}")
    print(f"  p50:     {p50:.2f}ms {'✅' if p50 < 50 else '⚠️'}")
    print(f"  p90:     {p90:.2f}ms")
    print(f"  avg:     {avg:.2f}ms")

print(f"\n{'='*80}")
print("Note: 768-dim is 6× larger than SIFT (128-dim)")
print("Expected latency: 2-3× slower than SIFT on Mac")
print("="*80)
