"""
Test Cohere 1M without quantization - Recall@100
"""
import numpy as np
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_DIR = '/Users/viktari/cohere_data'
INDEX_DIR = '/tmp/cohere_noquant'
NUM_QUERIES = 1000  # All queries available

print("="*80)
print("Cohere 1M - No Quantization - Recall@100")
print("="*80)

print("\nLoading data...")
base = np.fromfile(f'{DATA_DIR}/cohere_base.bin', dtype=np.float32)[:768000000].reshape(-1, 768)
queries = np.fromfile(f'{DATA_DIR}/cohere_query.bin', dtype=np.float32)[:768*NUM_QUERIES].reshape(-1, 768)
# Skip first 2 values (metadata: num_queries, k)
groundtruth = np.fromfile(f'{DATA_DIR}/cohere_groundtruth.bin', dtype=np.int32)[2:2+100*NUM_QUERIES].reshape(-1, 100)

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

if os.path.exists(os.path.join(INDEX_DIR, 'metadata.pkl')):
    print("\nLoading existing index...")
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
else:
    print("\nBuilding index...")
    sys.stdout.flush()
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
    sys.stdout.flush()

print("\nWarming up...")
for q in queries[:20]:
    index.search(q, base, k=100, search_internal_result_num=1024, max_check=49152)

print(f"\nTesting ({NUM_QUERIES} queries, k=100)...")
recalls = []
latencies = []

for i, query in enumerate(queries):
    t0 = time.perf_counter()
    indices, dists = index.search(
        query, base, k=100, 
        search_internal_result_num=1024,
        max_check=49152
    )
    latencies.append((time.perf_counter() - t0) * 1000)
    
    gt = set(groundtruth[i][:100])
    found = set(indices[:100])
    recalls.append(len(gt & found) / 100)
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{NUM_QUERIES}... (avg recall so far: {np.mean(recalls)*100:.2f}%)")

avg_recall = np.mean(recalls) * 100
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
avg = np.mean(latencies)

print(f"\n{'='*80}")
print("Results (No Quantization, Recall@100):")
print(f"  Recall:  {avg_recall:.2f}%")
print(f"  p50:     {p50:.2f}ms")
print(f"  p90:     {p90:.2f}ms")
print(f"  avg:     {avg:.2f}ms")
print("="*80)
