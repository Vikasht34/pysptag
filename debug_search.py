"""Debug search issue"""
import numpy as np
import h5py
import sys
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

DATA_FILE = 'data/cohere/documents-1m.hdf5'
INDEX_DIR = '/tmp/cohere_noquant'

print("Loading data...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:1]

print("Loading index...")
index = SPANNDiskOptimized(
    dim=768,
    use_rabitq=False,
    metric='IP',
    disk_path=INDEX_DIR,
    cache_size=2000
)

import pickle
with open(f'{INDEX_DIR}/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    for k, v in metadata.items():
        if k not in ['use_faiss_centroids', '_centroid_index', '_shared_rabitq']:
            setattr(index, k, v)

import faiss
index._centroid_index = faiss.IndexFlatIP(index.dim)
index._centroid_index.add(index.centroids.astype(np.float32))

print(f"\nIndex: {index.num_clusters} clusters, metric={index.metric}")
print(f"Query shape: {queries[0].shape}")

# Test search
print("\nSearching...")
dists, indices = index.search(
    queries[0], base, k=10,
    search_internal_result_num=48,
    max_check=6144
)

print(f"\nResults: {len(indices)} found")
print(f"Indices: {indices}")
print(f"Dists: {dists}")
