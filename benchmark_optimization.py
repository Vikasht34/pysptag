"""Benchmark optimized vs original RaBitQ"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.quantization.rabitq import RaBitQ
from src.quantization.rabitq_optimized import RaBitQOptimized

np.random.seed(42)
data = np.random.randn(10000, 128).astype(np.float32)
queries = np.random.randn(100, 128).astype(np.float32)

print("=" * 80)
print("BENCHMARK: Original vs Optimized RaBitQ")
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
    
    # Optimized
    rabitq_opt = RaBitQOptimized(dim=128, bq=bq, metric='L2')
    codes_opt = rabitq_opt.build(data)
    
    start = time.time()
    for query in queries:
        rabitq_opt.search(query, codes_opt, k=10)
    time_opt = time.time() - start
    
    # Verify correctness
    dists_orig, indices_orig = rabitq_orig.search(queries[0], codes_orig, data, k=10)
    dists_opt, indices_opt = rabitq_opt.search(queries[0], codes_opt, k=10)
    
    recall = len(set(indices_orig) & set(indices_opt)) / 10
    
    print(f"  Original:  {time_orig*1000:.2f}ms ({time_orig*10:.2f}ms/query)")
    print(f"  Optimized: {time_opt*1000:.2f}ms ({time_opt*10:.2f}ms/query)")
    print(f"  Speedup:   {time_orig/time_opt:.2f}×")
    print(f"  Recall:    {recall:.1%} (correctness check)")

print("\n" + "=" * 80)
