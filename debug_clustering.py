"""Check if clustering is working correctly"""
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

print(f"Index: {index.num_clusters} clusters")

# Check which clusters the ground truth vectors are in
print("\nChecking ground truth vector cluster assignments...")

# Find which cluster each GT vector belongs to
for i, gt_idx in enumerate(gt):
    # Find which cluster contains this vector
    found_in_clusters = []
    for cluster_id in range(min(100, index.num_clusters)):  # Check first 100 clusters
        posting = index._load_posting_mmap(cluster_id)
        if posting[0] is not None:
            posting_ids = posting[0]
            if gt_idx in posting_ids:
                found_in_clusters.append(cluster_id)
    
    if found_in_clusters:
        print(f"  GT[{i}] = {gt_idx} found in clusters: {found_in_clusters}")
    else:
        print(f"  GT[{i}] = {gt_idx} not found in first 100 clusters")

# Now check which centroids are nearest to the query
print("\nFinding nearest centroids to query...")
import faiss
if not hasattr(index, '_centroid_index') or index._centroid_index is None:
    # Build centroid index
    if index.metric == 'IP':
        index._centroid_index = faiss.IndexFlatIP(index.dim)
    else:
        index._centroid_index = faiss.IndexFlatL2(index.dim)
    index._centroid_index.add(index.centroids.astype(np.float32))

centroid_dists, nearest_centroids = index._centroid_index.search(
    query.reshape(1, -1).astype(np.float32), 
    48
)
print(f"Nearest 48 centroids: {nearest_centroids[0][:10]}...")

# Check if GT vectors are in the nearest centroids
print("\nChecking if GT vectors are in nearest centroids...")
for cluster_id in nearest_centroids[0][:48]:
    posting = index._load_posting_mmap(cluster_id)
    if posting[0] is not None:
        posting_ids = posting[0]
        gt_in_posting = [g for g in gt if g in posting_ids]
        if gt_in_posting:
            print(f"  Cluster {cluster_id}: contains GT vectors {gt_in_posting}")
