"""Quick SIFT 1M test - build + 100 queries."""
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

# Build index with RNG filtering
print("\nBuilding index...")
index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    tree_type='BKT',
    replica_count=8,
    use_rng_filtering=True,  # Enable RNG
    clustering='hierarchical',
    target_posting_size=800,  # Balance between quality and capacity
    disk_path='/tmp/sift1m_test',
    use_rabitq=False,
    preload_postings=True
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"✓ Build time: {build_time:.1f}s")

# Search on 100 queries
print("\nSearching 100 queries...")
k = 10
recalls = []
latencies = []

for i in range(100):
    q = queries[i]
    t0 = time.time()
    ids, _ = index.search(q, base, k=k, max_check=8192)
    latency = (time.time() - t0) * 1000
    latencies.append(latency)
    
    # Compute recall
    gt = groundtruth[i][:k]
    recall = len(set(ids) & set(gt)) / k
    recalls.append(recall)

# Results
print(f"\n{'='*50}")
print(f"SIFT 1M Quick Test (100 queries)")
print('='*50)
print(f"Build time: {build_time:.1f}s")
print(f"Clusters: {index.num_clusters} ({index.num_clusters/1000000*100:.2f}%)")
print(f"Recall@{k}: {np.mean(recalls)*100:.2f}%")
print(f"Latency: p50={np.percentile(latencies, 50):.2f}ms, p90={np.percentile(latencies, 90):.2f}ms, p99={np.percentile(latencies, 99):.2f}ms")
