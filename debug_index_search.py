"""Debug full index search"""
import numpy as np
import h5py
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_FILE = 'data/cohere/documents-1m.hdf5'
INDEX_DIR = '/tmp/cohere_1bit'

print("Loading data...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    query = f['test'][0]
    gt = f['neighbors'][0][:10]

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

print(f"Index: {index.num_clusters} clusters, metric={index.metric}")

# Search
print("\nSearching...")
indices, dists = index.search(query, base, k=10, search_internal_result_num=48, max_check=6144)

print(f"Top 10 indices: {indices}")
print(f"Top 10 dists: {dists}")
print(f"Ground truth: {gt}")

overlap = len(set(indices) & set(gt))
print(f"\nRecall: {overlap}/10 = {overlap*10}%")

# Check true distances
print("\nTrue distances for returned indices:")
true_dists = -np.dot(base[indices], query)
for i in range(len(indices)):
    print(f"  idx={indices[i]}, rabitq_dist={dists[i]:.4f}, true_dist={true_dists[i]:.4f}")

print("\nTrue distances for ground truth:")
gt_dists = -np.dot(base[gt], query)
for i in range(len(gt)):
    print(f"  idx={gt[i]}, true_dist={gt_dists[i]:.4f}")
