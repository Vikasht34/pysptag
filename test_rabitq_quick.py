"""Quick test of 2-bit quantization"""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.quantization.rabitq_proper import RaBitQ

# Small test
data = np.random.randn(100, 768).astype(np.float32)
query = np.random.randn(768).astype(np.float32)

print("Testing 2-bit RaBitQ...")
rabitq = RaBitQ(dim=768, bq=2, metric='IP')

print("Building...")
codes = rabitq.build(data)
print(f"Codes shape: {codes.shape}")
print(f"f_add shape: {rabitq.f_add.shape}")
print(f"f_rescale shape: {rabitq.f_rescale.shape}")

print("Searching...")
dists, indices = rabitq.search(query, codes, k=10)
print(f"Found {len(indices)} results")
print(f"Distances: {dists}")
print(f"Indices: {indices}")
print("✓ Works!")
