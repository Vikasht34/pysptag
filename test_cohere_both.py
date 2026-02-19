"""
Test both 1-bit and 2-bit quantization
"""
import numpy as np
import h5py
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_FILE = 'data/cohere/documents-1m.hdf5'
NUM_QUERIES = 100

print("="*80)
print("Cohere 1M - RaBitQ Quantization Test")
print("="*80)

print("\nLoading Cohere 1M...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

for bq, name in [(1, '1-bit'), (2, '2-bit')]:
    INDEX_DIR = f'/tmp/cohere_{bq}bit'
    
    print(f"\n{'='*80}")
    print(f"Testing {name} quantization")
    print("="*80)
    
    if os.path.exists(os.path.join(INDEX_DIR, 'metadata.pkl')):
        print(f"Loading existing {name} index...")
        index = SPANNDiskOptimized(
            dim=768,
            use_rabitq=True,
            bq=bq,
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
        print(f"Building {name} index...")
        t0 = time.time()
        index = SPANNDiskOptimized(
            dim=768,
            target_posting_size=500,
            replica_count=8,
            use_rabitq=True,
            bq=bq,
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
    
    print("Warming up...")
    for q in queries[:20]:
        index.search(q, base, k=10, search_internal_result_num=48, max_check=6144)
    
    print(f"Testing ({NUM_QUERIES} queries)...")
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
    
    print(f"\nResults ({name}):")
    print(f"  Recall:  {avg_recall:.2f}%")
    print(f"  p50:     {p50:.2f}ms")
    print(f"  p90:     {p90:.2f}ms")
    print(f"  avg:     {avg:.2f}ms")

print(f"\n{'='*80}")
print("Done")
print("="*80)
