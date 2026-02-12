"""Test SPANN with RaBitQ quantized posting lists"""
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.index.spann_rabitq import SPANNRaBitQ

print("="*80)
print("SPANN + RaBitQ (C++ Parameters: 64 centroids, 4096 maxCheck)")
print("="*80)

# Generate data
np.random.seed(42)
n_train = 100000
n_test = 100
dim = 128

print(f"\nGenerating {n_train:,} vectors, dim={dim}...")
train_data = np.random.randn(n_train, dim).astype(np.float32)
test_data = np.random.randn(n_test, dim).astype(np.float32)

# Build index
print("\nBuilding SPANN with quantized postings...")
t0 = time.time()
index = SPANNRaBitQ(dim=dim, target_posting_size=1000, bq=4)
index.build(train_data)
build_time = time.time() - t0

print(f"\n✓ Build time: {build_time:.2f}s")
print(f"  Clusters: {index.num_clusters}")
print(f"  Posting sizes: min={min(len(p) for p in index.posting_lists)}, "
      f"max={max(len(p) for p in index.posting_lists)}")

# Search with C++ parameters
print(f"\nSearching {n_test} queries (search_internal_result_num=64)...")
t0 = time.time()
recalls = []

for i in range(n_test):
    query = test_data[i]
    # Use C++ SPTAG default parameters
    dists, indices = index.search(query, train_data, k=10, 
                                  search_internal_result_num=64, 
                                  max_check=4096)
    
    # Compute recall
    true_dists = np.linalg.norm(train_data - query, axis=1)
    true_top10 = np.argsort(true_dists)[:10]
    recall = len(set(indices) & set(true_top10)) / 10
    recalls.append(recall)

search_time = time.time() - t0
qps = n_test / search_time

print(f"✓ Search time: {search_time:.2f}s")
print(f"  QPS: {qps:.2f}")
print(f"  Recall@10: {np.mean(recalls):.2%}")

print("\n" + "="*80)
print("✓ SPANN+RaBitQ TEST COMPLETE (C++ Parameters)")
print("="*80)

