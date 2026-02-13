"""Test with SIFT 1M dataset."""
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

def load_ivecs(filename):
    with open(filename, 'rb') as f:
        data = []
        while True:
            d = np.fromfile(f, dtype=np.int32, count=1)
            if len(d) == 0:
                break
            vec = np.fromfile(f, dtype=np.int32, count=d[0])
            data.append(vec)
    return np.array(data)

# Load SIFT 1M
print("Loading SIFT 1M...")
base = load_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs')
queries = load_fvecs('/Users/viktari/pysptag/data/sift/sift_query.fvecs')
groundtruth = load_ivecs('/Users/viktari/pysptag/data/sift/sift_groundtruth.ivecs')

print(f"Base: {base.shape}, Queries: {queries.shape}")

# Build index
print("\nBuilding index with BKTree + RNG...")
index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    tree_type='BKT',
    replica_count=8,
    use_rng_filtering=True,  # Enable RNG filtering
    clustering='hierarchical',
    target_posting_size=118
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"Build time: {build_time:.2f}s")

# Search
print("\nSearching...")
k = 10
recalls = []
latencies = []

for i, q in enumerate(queries):
    t0 = time.time()
    ids, _ = index.search(q, base, k=k)
    latency = (time.time() - t0) * 1000
    latencies.append(latency)
    
    # Compute recall
    gt = groundtruth[i][:k]
    recall = len(set(ids) & set(gt)) / k
    recalls.append(recall)
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(queries)} queries...")

# Results
print(f"\n{'='*50}")
print(f"SIFT 1M Results (BKTree + RNG)")
print('='*50)
print(f"Recall@{k}: {np.mean(recalls)*100:.2f}%")
print(f"Latency: p50={np.percentile(latencies, 50):.2f}ms, p90={np.percentile(latencies, 90):.2f}ms, p99={np.percentile(latencies, 99):.2f}ms")
print(f"Build time: {build_time:.2f}s")
