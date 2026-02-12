"""Fast SIFT1M test with optimized clustering"""
import numpy as np
import struct
import time
import sys
import os
sys.path.insert(0, os.path.expanduser('~/pysptag'))

def read_fvecs(filename, max_vecs=None):
    """Read .fvecs file format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
            if max_vecs and len(vectors) >= max_vecs:
                break
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename):
    """Read .ivecs file format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)

def fast_kmeans(data, k, max_iter=20):
    """Fast k-means with fewer iterations"""
    n = len(data)
    
    # Initialize with random samples
    idx = np.random.choice(n, k, replace=False)
    centroids = data[idx].copy()
    
    for iteration in range(max_iter):
        # Assignment (vectorized)
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # Check convergence
        diff = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids
        
        if diff < 1e-4:
            break
    
    return centroids, labels

print("="*80)
print("SIFT1M Fast Test - Optimized Clustering")
print("="*80)

# Paths
data_dir = os.path.expanduser('~/pysptag/data/sift')

# Load SIFT1M
print("\n[1/4] Loading SIFT1M dataset...")
t0 = time.time()
base = read_fvecs(f'{data_dir}/sift_base.fvecs')
queries = read_fvecs(f'{data_dir}/sift_query.fvecs')
groundtruth = read_ivecs(f'{data_dir}/sift_groundtruth.ivecs')
print(f"✓ Loaded in {time.time()-t0:.1f}s - Base: {base.shape}, Queries: {queries.shape}")

# Fast clustering
print("\n[2/4] Fast k-means clustering...")
t0 = time.time()
n_clusters = 100
centroids, labels = fast_kmeans(base, n_clusters, max_iter=20)
print(f"✓ Clustered in {time.time()-t0:.1f}s - {n_clusters} clusters")

# Build posting lists
print("\n[3/4] Building posting lists...")
t0 = time.time()
posting_lists = [[] for _ in range(n_clusters)]
for i, label in enumerate(labels):
    posting_lists[label].append(i)

# Convert to numpy arrays
posting_lists = [np.array(p, dtype=np.int32) for p in posting_lists]
sizes = [len(p) for p in posting_lists]
print(f"✓ Built in {time.time()-t0:.1f}s - Avg size: {np.mean(sizes):.0f}, Min: {min(sizes)}, Max: {max(sizes)}")

# Search
print("\n[4/4] Searching...")
t0 = time.time()
n_probe = 10
recalls = []

for i, query in enumerate(queries):
    # Find nearest centroids
    dists = np.sum((query - centroids) ** 2, axis=1)
    nearest = np.argpartition(dists, n_probe)[:n_probe]
    
    # Get candidates from posting lists
    candidates = []
    for c in nearest:
        candidates.extend(posting_lists[c])
    
    if not candidates:
        continue
    
    # Compute distances
    candidates = list(set(candidates))[:1000]  # Limit candidates
    candidate_vecs = base[candidates]
    dists = np.sum((query - candidate_vecs) ** 2, axis=1)
    
    # Get top-10
    top_k = np.argpartition(dists, min(10, len(dists)-1))[:10]
    result_ids = [candidates[j] for j in top_k]
    
    # Compute recall@10
    gt = set(groundtruth[i][:10])
    found = set(result_ids)
    recalls.append(len(gt & found) / 10)
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(queries)} queries - Recall@10: {np.mean(recalls):.2%}")

search_time = time.time() - t0
print(f"\n✓ Search complete: {search_time:.1f}s, QPS: {len(queries)/search_time:.1f}")
print(f"\n📊 Final Recall@10: {np.mean(recalls):.2%}")
print("="*80)
