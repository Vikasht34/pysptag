"""
Rebuild Cohere index with HNSW and verify it works
"""
import sys
import numpy as np
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

print("Loading Cohere data...")
base = np.fromfile('/Users/viktari/cohere_data/cohere_base.bin', dtype=np.float32)[2:].reshape(-1, 768)
queries = np.fromfile('/Users/viktari/cohere_data/cohere_query.bin', dtype=np.float32)[2:].reshape(-1, 768)
gt_raw = np.fromfile('/Users/viktari/cohere_data/cohere_groundtruth.bin', dtype=np.int32)[2:]
groundtruth = gt_raw[:len(gt_raw) // 100 * 100].reshape(-1, 100)
print(f"Loaded: base={base.shape}, queries={queries.shape}, GT={groundtruth.shape}")

print("\nBuilding index with HNSW...")
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,
    metric='IP',
    use_hnsw_centroids=True,
    hnsw_m=16,
    hnsw_ef_construction=200,
    clustering='hierarchical',
    use_rng_filtering=True,
    use_faiss_centroids=False,
    disk_path='/tmp/cohere_index_new',
    cache_size=2000
)

index.build(base)
print(f"✓ Index built: {index.num_clusters} clusters")

# Verify vector 278491 is in the index
print("\nVerifying index integrity...")
true_nn = 278491
vec = base[true_nn]
centroid_scores = np.dot(index.centroids, vec)
nearest_centroid = np.argmax(centroid_scores)
print(f"Vector {true_nn} should be in centroid {nearest_centroid}")

posting = index._load_posting_mmap(nearest_centroid)
if posting and true_nn in posting[0]:
    print(f"✓ Vector {true_nn} IS in centroid {nearest_centroid}'s posting list")
else:
    print(f"✗ Vector {true_nn} is NOT in posting list - INDEX IS BROKEN!")
    sys.exit(1)

# Test search
print("\nTesting search...")
query = queries[0]
indices, dists = index.search(query, base, k=10, search_internal_result_num=32, max_check=4096)
print(f"Returned: {indices[:10]}")
print(f"GT:       {groundtruth[0][:10]}")
overlap = len(set(indices[:10]) & set(groundtruth[0][:10]))
print(f"Overlap: {overlap}/10")

if overlap > 0:
    print("\n✓ Index works!")
else:
    print("\n✗ Index broken - 0 overlap")
