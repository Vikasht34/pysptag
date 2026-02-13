"""Quick 1M test with minimal memory"""
import numpy as np
import time
from src.utils.io import load_fvecs

print("Loading SIFT 1M base...")
base = load_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs')
print(f"Loaded: {base.shape}, Memory: {base.nbytes/1024**3:.2f} GB")

# Simple clustering without BKTree
print("\nClustering with simple k-means...")
from src.clustering.kmeans import KMeansClustering

clusterer = KMeansClustering(metric='L2')
t0 = time.time()
centroids, labels = clusterer.cluster(base, target_clusters=10000)
print(f"Clustered in {time.time()-t0:.1f}s: {len(centroids)} clusters")

# Simple assignment without RNG
print("\nAssigning vectors (no RNG)...")
from src.clustering.hierarchical import HierarchicalClustering
hc = HierarchicalClustering()
t0 = time.time()
postings, counts = hc.assign_with_replicas(
    base, centroids, replica_count=8, posting_limit=500, use_rng_filtering=False
)
print(f"Assigned in {time.time()-t0:.1f}s")

sizes = [len(p) for p in postings if len(p) > 0]
print(f"Posting sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
print(f"Avg replicas: {counts.mean():.2f}")
print("\n✓ Success - no hang, reasonable memory")
