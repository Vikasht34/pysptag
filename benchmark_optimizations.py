"""
Benchmark the 3 new optimizations:
1. Posting page limit
2. Batch query processing
3. Hash table deduplication
"""
import numpy as np
import pickle
import time
import struct
import faiss
from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

def load_ivecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            data.append(vec)
    return np.array(data, dtype=np.int32)

print("Loading data...")
queries = load_fvecs('data/sift/sift_query.fvecs')
base = load_fvecs('data/sift/sift_base.fvecs')
ground_truth = load_ivecs('data/sift/sift_groundtruth.ivecs')

# Load index
index = SPANNDiskOptimized(
    dim=128, metric='L2',
    disk_path='/tmp/sift1m_final_500',
    use_rabitq=True,
    use_faiss_centroids=True,
    cache_size=2000
)

with open('/tmp/sift1m_final_500/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    for k, v in metadata.items():
        if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq']:
            setattr(index, k, v)

index._centroid_index = faiss.IndexFlatL2(index.dim)
index._centroid_index.add(index.centroids.astype(np.float32))

# Test configurations
configs = [
    {
        'name': 'Baseline (async pruning)',
        'use_async_pruning': True,
        'max_vectors_per_posting': None,
        'use_batch': False
    },
    {
        'name': 'With posting limit (500)',
        'use_async_pruning': True,
        'max_vectors_per_posting': 500,
        'use_batch': False
    },
    {
        'name': 'With posting limit (300)',
        'use_async_pruning': True,
        'max_vectors_per_posting': 300,
        'use_batch': False
    },
    {
        'name': 'Batch queries (no limit)',
        'use_async_pruning': True,
        'max_vectors_per_posting': None,
        'use_batch': True
    },
    {
        'name': 'All optimizations (limit=300)',
        'use_async_pruning': True,
        'max_vectors_per_posting': 300,
        'use_batch': True
    }
]

num_queries = 1000

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print("="*70)
    
    latencies = []
    recalls = []
    
    if config['use_batch']:
        # Batch processing
        batch_size = 100
        for i in range(0, num_queries, batch_size):
            batch_queries = queries[i:i+batch_size]
            
            t0 = time.perf_counter()
            ids_batch, _ = index.search_batch(
                batch_queries, base, k=10,
                search_internal_result_num=48,
                max_check=6144,
                use_async_pruning=config['use_async_pruning'],
                max_vectors_per_posting=config['max_vectors_per_posting']
            )
            batch_time = (time.perf_counter() - t0) * 1000
            
            # Record per-query latency
            for j in range(len(batch_queries)):
                latencies.append(batch_time / len(batch_queries))
                
                gt = set(ground_truth[i+j, :10])
                pred = set(ids_batch[j, :10])
                recalls.append(len(gt & pred) / 10.0)
            
            if (i + batch_size) % 200 == 0:
                print(f"  {i + batch_size}/{num_queries}")
    else:
        # Single query
        for i in range(num_queries):
            t0 = time.perf_counter()
            ids, _ = index.search(
                queries[i], base, k=10,
                search_internal_result_num=48,
                max_check=6144,
                use_async_pruning=config['use_async_pruning'],
                max_vectors_per_posting=config['max_vectors_per_posting']
            )
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)
            
            gt = set(ground_truth[i, :10])
            pred = set(ids[:10])
            recalls.append(len(gt & pred) / 10.0)
            
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{num_queries}")
    
    mean_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    avg = np.mean(latencies)
    
    print(f"\nResults:")
    print(f"  Recall:  {mean_recall:.2f}% {'✅' if mean_recall >= 90 else '⚠️'}")
    print(f"  p50:     {p50:.2f}ms {'✅' if p50 < 10 else '⚠️'}")
    print(f"  p90:     {p90:.2f}ms")
    print(f"  avg:     {avg:.2f}ms")

print(f"\n{'='*70}")
print("Summary:")
print("  1. Posting limit: Reduces vectors checked per posting")
print("  2. Batch queries: Amortizes overhead across queries")
print("  3. Hash dedup: Fast deduplication (always enabled)")
