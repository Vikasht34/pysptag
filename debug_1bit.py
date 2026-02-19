"""Debug 1-bit vs no-quant"""
import numpy as np
import h5py
import sys
sys.path.insert(0, '.')
from src.quantization.rabitq_proper import RaBitQ

DATA_FILE = 'data/cohere/documents-1m.hdf5'

print("Loading data...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:1000]  # Small sample
    query = f['test'][0]
    gt = f['neighbors'][0][:10]

print(f"Base: {base.shape}, Query: {query.shape}")

# Test 1-bit quantization
print("\n1-bit quantization:")
rabitq = RaBitQ(dim=768, bq=1, metric='IP')
codes = rabitq.build(base)
print(f"Codes shape: {codes.shape}")
print(f"Packed dim: {codes.shape[1]} (should be {(768+7)//8})")

# Search
dists, indices = rabitq.search(query, codes, k=10)
print(f"Top 10 indices: {indices}")
print(f"Top 10 dists: {dists}")

# Compare with true IP
print("\nTrue IP distances:")
true_dists = -np.dot(base, query)
true_indices = np.argsort(true_dists)[:10]
print(f"True top 10 indices: {true_indices}")
print(f"True top 10 dists: {true_dists[true_indices]}")

# Check overlap
overlap = len(set(indices) & set(true_indices))
print(f"\nOverlap: {overlap}/10 = {overlap*10}% recall")

# Check if distances are in right order
print("\nCompare distance ordering:")
for i in range(min(5, len(indices))):
    idx = indices[i]
    print(f"  Rank {i}: idx={idx}, rabitq_dist={dists[i]:.4f}, true_dist={true_dists[idx]:.4f}")
