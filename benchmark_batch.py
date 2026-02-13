"""Benchmark batch query processing"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from src.index.spann_rabitq_replica import SPANNRaBitQReplica
from src.index.batch_search import search_batch

np.random.seed(42)
data = np.random.randn(100000, 128).astype(np.float32)
queries = np.random.randn(100, 128).astype(np.float32)

print("Building index...")
index = SPANNRaBitQReplica(dim=128, target_posting_size=500, replica_count=4, bq=2, metric='L2')
index.build(data)

print("\n" + "="*80)
print("BENCHMARK: Single vs Batch Query Processing")
print("="*80)

# Single query
print("\nSingle query processing:")
start = time.time()
for query in queries:
    dists, indices = index.search(query, data, k=10, search_internal_result_num=10, max_check=200)
time_single = time.time() - start
print(f"  Time: {time_single*1000:.2f}ms ({time_single*10:.2f}ms/query)")
print(f"  QPS: {len(queries)/time_single:.1f}")

# Batch query
print("\nBatch query processing:")
start = time.time()
results = search_batch(index, queries, data, k=10, search_internal_result_num=10, max_check=200)
time_batch = time.time() - start
print(f"  Time: {time_batch*1000:.2f}ms ({time_batch*10:.2f}ms/query)")
print(f"  QPS: {len(queries)/time_batch:.1f}")

print(f"\nSpeedup: {time_single/time_batch:.2f}×")

# Verify correctness
dists_single, indices_single = index.search(queries[0], data, k=10, search_internal_result_num=10, max_check=200)
dists_batch, indices_batch = results[0]
recall = len(set(indices_single) & set(indices_batch)) / 10
print(f"Recall (correctness check): {recall:.1%}")

print("="*80)
