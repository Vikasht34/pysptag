"""Minimal end-to-end test for all metric types."""
import numpy as np
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Generate small test data
np.random.seed(42)
data = np.random.randn(1000, 32).astype(np.float32)
queries = np.random.randn(10, 32).astype(np.float32)

# Normalize for IP/Cosine
data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)

metrics = ['L2', 'IP', 'Cosine']

for metric in metrics:
    print(f"\n{'='*50}")
    print(f"Testing {metric}")
    print('='*50)
    
    # Use normalized data for IP/Cosine
    test_data = data_norm if metric in ['IP', 'Cosine'] else data
    test_queries = queries_norm if metric in ['IP', 'Cosine'] else queries
    
    # Build index
    index = SPANNDiskOptimized(
        dim=32,
        metric=metric,
        replica_count=3,
        use_rng_filtering=True
    )
    
    index.build(test_data)
    
    # Search
    results = []
    for q in test_queries:
        ids, dists = index.search(q, test_data, k=5)
        results.append(ids)
    
    print(f"✓ Built {len(test_data)} vectors, searched {len(test_queries)} queries")
    print(f"  Sample distances: {dists[:3]}")

print(f"\n{'='*50}")
print("✓ All metrics passed!")
print('='*50)
