"""
Quick test: Verify IP metric works correctly
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

# Create small test data
np.random.seed(42)
n = 1000
dim = 128
data = np.random.randn(n, dim).astype(np.float32)
data = data / np.linalg.norm(data, axis=1, keepdims=True)  # Normalize for IP

# Build index
print("Building index with Cosine metric...")
index = SPANNRaBitQReplica(
    dim=dim,
    target_posting_size=100,
    replica_count=4,
    bq=1,
    use_rabitq=True,
    metric='Cosine'
)
index.build(data)

# Test search
query = data[0]  # Use first vector as query
print("\nSearching...")
dists, indices = index.search(query, data, k=10)

# Compute ground truth
gt_dists = -np.dot(data, query)
gt_indices = np.argsort(gt_dists)[:10]

print(f"\nReturned indices: {indices[:5]}")
print(f"Ground truth indices: {gt_indices[:5]}")
print(f"Overlap: {len(set(indices) & set(gt_indices))}/10")

# Check if query itself is returned (should be index 0)
if 0 in indices:
    print(f"✓ Query found at position {np.where(indices == 0)[0][0]}")
else:
    print("✗ Query NOT found in results (BUG!)")
