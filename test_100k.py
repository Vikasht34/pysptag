"""
Test PySPTAG on 100K vectors with 5K queries
"""
import numpy as np
import time
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.core.bktree import BKTree
from src.index.spann_exact import SPANNExact
from src.quantization.rabitq import RaBitQ

print("="*80)
print("PySPTAG - 100K Vectors + 5K Queries Test")
print("="*80)

# Generate data
np.random.seed(42)
n_train = 100000
n_test = 5000
dim = 128

print(f"\nGenerating data...")
train_data = np.random.randn(n_train, dim).astype(np.float32)
test_data = np.random.randn(n_test, dim).astype(np.float32)
print(f"✓ Dataset: {n_train:,} train, {n_test:,} test, dim={dim}")

# Test 1: SPANN with BKTree+RNG
print("\n" + "="*80)
print("Test 1: SPANN with BKTree+RNG")
print("="*80)

t0 = time.time()
spann = SPANNExact(dim=dim, target_posting_size=1000, num_trees=1, kmeans_k=32)
spann.build(train_data)
build_time = time.time() - t0

print(f"\n✓ Build complete: {build_time:.2f}s")
print(f"  Clusters: {spann.num_clusters}")
print(f"  Posting sizes: min={min(len(p) for p in spann.posting_lists)}, "
      f"max={max(len(p) for p in spann.posting_lists)}, "
      f"avg={np.mean([len(p) for p in spann.posting_lists]):.1f}")

# Search
print(f"\nSearching {n_test:,} queries...")
t0 = time.time()
recalls = []
for i in range(min(100, n_test)):  # Test first 100 queries
    query = test_data[i]
    dists, indices = spann.search(query, train_data, k=10, num_postings=20)
    
    # Compute recall
    true_dists = np.linalg.norm(train_data - query, axis=1)
    true_top10 = np.argsort(true_dists)[:10]
    recall = len(set(indices) & set(true_top10)) / 10
    recalls.append(recall)

search_time = time.time() - t0
qps = 100 / search_time

print(f"✓ Search complete: {search_time:.2f}s")
print(f"  QPS: {qps:.2f}")
print(f"  Recall@10: {np.mean(recalls):.2%}")

# Test 2: RaBitQ
print("\n" + "="*80)
print("Test 2: RaBitQ Quantization")
print("="*80)

t0 = time.time()
rabitq = RaBitQ(dim=dim, bq=4)
codes = rabitq.build(train_data)
build_time = time.time() - t0

print(f"\n✓ Build complete: {build_time:.2f}s")
print(f"  Codes shape: {codes.shape}")
print(f"  Compression: {rabitq.get_compression_ratio():.1f}×")
print(f"  Original: {train_data.nbytes / 1024 / 1024:.2f} MB")
print(f"  Compressed: {codes.nbytes / 1024 / 1024:.2f} MB")
print(f"  Savings: {(1 - codes.nbytes / train_data.nbytes) * 100:.1f}%")

# Search
print(f"\nSearching {n_test:,} queries...")
t0 = time.time()
recalls = []
for i in range(min(100, n_test)):
    query = test_data[i]
    dists, indices = rabitq.search(query, codes, train_data, k=10)
    
    # Compute recall
    true_dists = np.linalg.norm(train_data - query, axis=1)
    true_top10 = np.argsort(true_dists)[:10]
    recall = len(set(indices) & set(true_top10)) / 10
    recalls.append(recall)

search_time = time.time() - t0
qps = 100 / search_time

print(f"✓ Search complete: {search_time:.2f}s")
print(f"  QPS: {qps:.2f}")
print(f"  Recall@10: {np.mean(recalls):.2%}")

print("\n" + "="*80)
print("✓ 100K TEST COMPLETE")
print("="*80)
