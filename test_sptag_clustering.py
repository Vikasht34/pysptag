"""
Test SPTAG-style dynamic clustering
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("SPTAG Dynamic Clustering Test")
print("="*80)

# Test data
np.random.seed(42)
data = np.random.randn(10000, 768).astype(np.float32)
queries = np.random.randn(100, 768).astype(np.float32)

# Ground truth
print("\nComputing ground truth...")
gt_indices = []
for query in queries:
    dists = np.sum((data - query) ** 2, axis=1)
    gt_indices.append(np.argsort(dists)[:10])
gt_indices = np.array(gt_indices)

# Test 1: Original k-means clustering
print(f"\n{'='*80}")
print("Test 1: Original K-Means Clustering")
print("="*80)

index_kmeans = ProfiledSPANN(
    dim=768,
    target_posting_size=118,
    replica_count=8,
    bq=4,
    use_rabitq=False,
    tree_type='KDT',
    disk_path='./test_kmeans',
    cache_size=256
)

t0 = time.time()
index_kmeans.build(data, use_sptag_clustering=False)
build_time_kmeans = time.time() - t0
print(f"✓ Built in {build_time_kmeans:.2f}s")

# Warm up
for query in queries[:5]:
    index_kmeans.search(query, data, k=10, search_internal_result_num=64, max_check=8192)

# Test
index_kmeans.reset_stats()
recalls_kmeans = []
latencies_kmeans = []

for i, query in enumerate(queries[:50]):
    t0 = time.time()
    dists, indices = index_kmeans.search(query, data, k=10, 
                                         search_internal_result_num=64,
                                         max_check=8192)
    latencies_kmeans.append((time.time() - t0) * 1000)
    
    gt = set(gt_indices[i][:10])
    found = set(indices[:10])
    recalls_kmeans.append(len(gt & found) / 10)

p50_kmeans = np.percentile(latencies_kmeans, 50)
p90_kmeans = np.percentile(latencies_kmeans, 90)
recall_kmeans = np.mean(recalls_kmeans) * 100

print(f"\nResults:")
print(f"  Latency: p50={p50_kmeans:.2f}ms, p90={p90_kmeans:.2f}ms")
print(f"  Recall: {recall_kmeans:.1f}%")
print(f"  Clusters: {index_kmeans.num_clusters}")

# Test 2: SPTAG dynamic clustering
print(f"\n{'='*80}")
print("Test 2: SPTAG Dynamic Clustering")
print("="*80)

index_sptag = ProfiledSPANN(
    dim=768,
    target_posting_size=118,  # Used as posting limit
    replica_count=8,
    bq=4,
    use_rabitq=False,
    tree_type='BKT',  # SPTAG uses BKTree
    disk_path='./test_sptag',
    cache_size=256
)

t0 = time.time()
index_sptag.build(data, use_sptag_clustering=True)
build_time_sptag = time.time() - t0
print(f"✓ Built in {build_time_sptag:.2f}s")

# Warm up
for query in queries[:5]:
    index_sptag.search(query, data, k=10, search_internal_result_num=64, max_check=8192)

# Test
index_sptag.reset_stats()
recalls_sptag = []
latencies_sptag = []

for i, query in enumerate(queries[:50]):
    t0 = time.time()
    dists, indices = index_sptag.search(query, data, k=10, 
                                        search_internal_result_num=64,
                                        max_check=8192)
    latencies_sptag.append((time.time() - t0) * 1000)
    
    gt = set(gt_indices[i][:10])
    found = set(indices[:10])
    recalls_sptag.append(len(gt & found) / 10)

p50_sptag = np.percentile(latencies_sptag, 50)
p90_sptag = np.percentile(latencies_sptag, 90)
recall_sptag = np.mean(recalls_sptag) * 100

print(f"\nResults:")
print(f"  Latency: p50={p50_sptag:.2f}ms, p90={p90_sptag:.2f}ms")
print(f"  Recall: {recall_sptag:.1f}%")
print(f"  Clusters: {index_sptag.num_clusters}")

# Comparison
print(f"\n{'='*80}")
print("COMPARISON")
print("="*80)
print(f"{'Method':<20} {'Clusters':<12} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10}")
print("-"*80)
print(f"{'K-Means':<20} {index_kmeans.num_clusters:<12} {p50_kmeans:<10.2f} {p90_kmeans:<10.2f} {recall_kmeans:<10.1f}")
print(f"{'SPTAG Dynamic':<20} {index_sptag.num_clusters:<12} {p50_sptag:<10.2f} {p90_sptag:<10.2f} {recall_sptag:<10.1f}")
print("="*80)

if p90_sptag < p90_kmeans:
    improvement = (p90_kmeans - p90_sptag) / p90_kmeans * 100
    print(f"\n✅ SPTAG clustering is {improvement:.1f}% faster (p90)!")
else:
    print(f"\n⚠️  K-means is faster")

if recall_sptag > recall_kmeans:
    print(f"✅ SPTAG has {recall_sptag - recall_kmeans:.1f}% better recall!")
elif recall_sptag < recall_kmeans:
    print(f"⚠️  K-means has {recall_kmeans - recall_sptag:.1f}% better recall")

# Cleanup
import shutil
shutil.rmtree('./test_kmeans', ignore_errors=True)
shutil.rmtree('./test_sptag', ignore_errors=True)
