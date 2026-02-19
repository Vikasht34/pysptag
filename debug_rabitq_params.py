"""Check RaBitQ params loading"""
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

# Load a posting and check RaBitQ params
print("\nLoading cluster 6470 (contains GT vectors)...")
posting_ids, codes, rabitq = index._load_posting_mmap(6470)

print(f"Posting IDs shape: {posting_ids.shape}")
print(f"Codes shape: {codes.shape}")
print(f"RaBitQ type: {type(rabitq)}")
print(f"RaBitQ centroid shape: {rabitq.centroid.shape}")
print(f"RaBitQ f_add shape: {rabitq.f_add.shape}")
print(f"RaBitQ f_rescale shape: {rabitq.f_rescale.shape}")
print(f"RaBitQ metric: {rabitq.metric}")
print(f"RaBitQ bq: {rabitq.bq}")

# Check if GT vector 278491 is in this posting
gt_idx = 278491
if gt_idx in posting_ids:
    local_idx = np.where(posting_ids == gt_idx)[0][0]
    print(f"\nGT vector {gt_idx} is at local index {local_idx}")
    print(f"  f_add: {rabitq.f_add[local_idx]}")
    print(f"  f_rescale: {rabitq.f_rescale[local_idx]}")
    
    # Compute distance manually
    dists, indices = rabitq.search(query, codes, k=len(posting_ids))
    rank = np.where(indices == local_idx)[0][0]
    print(f"  RaBitQ distance: {dists[rank]}")
    print(f"  Rank in posting: {rank+1}/{len(posting_ids)}")
    
    # True distance
    true_dist = -np.dot(base[gt_idx], query)
    print(f"  True distance: {true_dist}")
    
    # Check top 10 in this posting
    print(f"\nTop 10 in this posting:")
    for i in range(min(10, len(indices))):
        global_idx = posting_ids[indices[i]]
        true_d = -np.dot(base[global_idx], query)
        print(f"    Rank {i+1}: global_idx={global_idx}, rabitq_dist={dists[i]:.4f}, true_dist={true_d:.4f}")
