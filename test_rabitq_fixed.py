"""Test fixed RaBitQ"""
import numpy as np
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.quantization.rabitq_fixed import RaBitQFixed

# Simple test
np.random.seed(42)
dim = 128
n = 100

# Create data
data = np.random.randn(n, dim).astype(np.float32)

# Build RaBitQ
rabitq = RaBitQFixed(dim=dim)
codes = rabitq.build(data)

print(f"Built RaBitQ:")
print(f"  Codes shape: {codes.shape}")

# Test search with first vector (should find itself)
query = data[0]
print(f"\nSearching for first vector...")
dists, indices = rabitq.search(query, codes, k=10)

print(f"\nResults:")
print(f"  Top-10 indices: {indices}")
print(f"  Top-10 distances: {dists}")
print(f"  Expected index 0 at position: {np.where(indices == 0)[0]}")

# Compute true distance
true_dist = np.linalg.norm(query - data[0])
print(f"\nTrue distance to itself: {true_dist:.6f}")
print(f"Estimated distance: {dists[0]:.6f}")

# Test on all vectors
print(f"\nTesting recall...")
recalls = []
for i in range(min(20, n)):
    query = data[i]
    dists, indices = rabitq.search(query, codes, k=10)
    
    # Should find itself in top-10
    if i in indices:
        recalls.append(1)
    else:
        recalls.append(0)

print(f"Self-recall: {np.mean(recalls):.2%} (should be 100%)")
