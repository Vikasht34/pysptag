"""
Benchmark HierarchicalClustering BKTree build time on 10M vectors
"""
import sys
import numpy as np
import time
sys.path.insert(0, '.')
from src.clustering.hierarchical import HierarchicalClustering

# Generate 10M random vectors (200d to match GloVe)
print("Generating 10M random vectors (200d)...")
np.random.seed(42)
data = np.random.randn(10_000_000, 200).astype(np.float32)
# Normalize for cosine
data = data / np.linalg.norm(data, axis=1, keepdims=True)
print(f"Generated: {data.shape}, {data.nbytes / 1024**3:.2f} GB")

print("\nBuilding HierarchicalClustering with BKTree...")
clusterer = HierarchicalClustering(
    select_threshold=0,
    split_threshold=0,
    split_factor=0,
    ratio=0.01,  # 1% = 100K centroids
    kmeans_k=32,
    leaf_size=8,
    metric='Cosine'
)

t0 = time.time()
centroids, labels = clusterer.cluster(data)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"{'='*60}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
print(f"Centroids: {len(centroids)} ({len(centroids)/len(data)*100:.2f}%)")
print(f"Throughput: {len(data)/elapsed:.0f} vectors/sec")
print(f"{'='*60}")
