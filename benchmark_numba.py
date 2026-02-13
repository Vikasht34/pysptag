"""Benchmark Numba JIT optimization"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.quantization.rabitq import RaBitQ
from src.quantization.rabitq_numba import RaBitQNumba

np.random.seed(42)
data = np.random.randn(10000, 128).astype(np.float32)
queries = np.random.randn(100, 128).astype(np.float32)

print("=" * 80)
print("BENCHMARK: Original vs Numba JIT")
print("=" * 80)

for bq in [1, 2, 4]:
    print(f"\n{bq}-bit quantization:")
    print("-" * 80)
    
    # Original
    rabitq_orig = RaBitQ(dim=128, bq=bq, metric='L2')
    codes_orig = rabitq_orig.build(data)
    
    start = time.time()
    for query in queries:
        rabitq_orig.search(query, codes_orig, data, k=10)
    time_orig = time.time() - start
    
    # Numba (with warmup)
    rabitq_numba = RaBitQNumba(dim=128, bq=bq, metric='L2')
    codes_numba = rabitq_numba.build(data)
    
    # Warmup JIT
    for _ in range(5):
        rabitq_numba.search(queries[0], codes_numba, k=10)
    
    start = time.time()
    for query in queries:
        rabitq_numba.search(query, codes_numba, k=10)
    time_numba = time.time() - start
    
    # Verify correctness
    dists_orig, indices_orig = rabitq_orig.search(queries[0], codes_orig, data, k=10)
    dists_numba, indices_numba = rabitq_numba.search(queries[0], codes_numba, k=10)
    
    recall = len(set(indices_orig) & set(indices_numba)) / 10
    
    print(f"  Original: {time_orig*1000:.2f}ms ({time_orig*10:.2f}ms/query)")
    print(f"  Numba:    {time_numba*1000:.2f}ms ({time_numba*10:.2f}ms/query)")
    print(f"  Speedup:  {time_orig/time_numba:.2f}×")
    print(f"  Recall:   {recall:.1%} (correctness check)")

print("\n" + "=" * 80)
print("Testing IP metric...")
print("=" * 80)

for bq in [1, 2]:
    rabitq_orig = RaBitQ(dim=128, bq=bq, metric='IP')
    codes_orig = rabitq_orig.build(data)
    
    rabitq_numba = RaBitQNumba(dim=128, bq=bq, metric='IP')
    codes_numba = rabitq_numba.build(data)
    
    # Warmup
    for _ in range(5):
        rabitq_numba.search(queries[0], codes_numba, k=10)
    
    start = time.time()
    for query in queries[:20]:
        rabitq_numba.search(query, codes_numba, k=10)
    time_numba = time.time() - start
    
    print(f"{bq}-bit IP: {time_numba*1000:.2f}ms for 20 queries ({time_numba*50:.2f}ms/query)")

print("\n" + "=" * 80)
