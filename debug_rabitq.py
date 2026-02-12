"""Debug RaBitQ distance estimation"""
import numpy as np
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.quantization.rabitq import RaBitQ

# Simple test: quantize and search
np.random.seed(42)
dim = 128
n = 100

# Create data
data = np.random.randn(n, dim).astype(np.float32)

# Build RaBitQ
rabitq = RaBitQ(dim=dim, bq=4)
codes = rabitq.build(data)

print(f"Built RaBitQ:")
print(f"  Centroid shape: {rabitq.centroid.shape}")
print(f"  Norms shape: {rabitq.norms.shape}")
print(f"  f_add shape: {rabitq.f_add.shape}")
print(f"  f_rescale shape: {rabitq.f_rescale.shape}")
print(f"  Codes shape: {codes.shape}")

# Test search with first vector (should find itself)
query = data[0]
print(f"\nSearching for first vector...")
dists, indices = rabitq.search(query, codes, data, k=10)

print(f"\nResults:")
print(f"  Top-10 indices: {indices}")
print(f"  Top-10 distances: {dists}")
print(f"  Expected index 0 at position: {np.where(indices == 0)[0]}")

# Compute true distance
true_dist = np.linalg.norm(query - data[0])
print(f"\nTrue distance to itself: {true_dist:.6f}")
print(f"Estimated distance: {dists[0] if len(dists) > 0 else 'N/A':.6f}")
