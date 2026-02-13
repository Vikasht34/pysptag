"""
Test balanced k-means impact on SIFT1M recall
"""
import numpy as np
import time
from src.index.spann_disk_optimized import SPANNDiskOptimized
from src.utils.io import load_fvecs

print("="*60)
print("SIFT1M: Balanced K-means Impact Test")
print("="*60)

# Load SIFT1M
print("\nLoading SIFT1M dataset...")
base = load_fvecs('data/sift/sift_base.fvecs')[:100000]  # Use 100K for faster testing
queries = load_fvecs('data/sift/sift_query.fvecs')[:100]
ground_truth = np.fromfile('data/sift/sift_groundtruth.ivecs', dtype=np.int32)
ground_truth = ground_truth.reshape(-1, 101)[:100, 1:11]  # Top 10

print(f"Base: {base.shape}")
print(f"Queries: {queries.shape}")

# Test 1: Standard k-means (current)
print("\n[Test 1] Standard K-means Clustering")
print("-"*60)

index1 = SPANNDiskOptimized(
    dim=128,
    target_posting_size=200,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    clustering='kmeans',  # Standard k-means
    disk_path='./test_index_standard'
)

print("Building index with standard k-means...")
start = time.time()
index1.build(base)
build_time1 = time.time() - start
print(f"Build time: {build_time1:.2f}s")

# Search
print("Searching...")
recalls = []
latencies = []

for i, query in enumerate(queries):
    start = time.time()
    dists, indices = index1.search(
        query, base, k=10,
        search_internal_result_num=64,
        max_check=4096
    )
    latency = (time.perf_counter() - start) * 1000
    latencies.append(latency)
    
    # Calculate recall
    recall = len(set(indices) & set(ground_truth[i])) / 10
    recalls.append(recall)

print(f"Recall@10: {np.mean(recalls)*100:.2f}%")
print(f"Latency p50: {np.percentile(latencies, 50):.2f}ms")
print(f"Latency p90: {np.percentile(latencies, 90):.2f}ms")

# Test 2: Hierarchical with balanced k-means
print("\n[Test 2] Hierarchical Clustering (Balanced K-means)")
print("-"*60)

index2 = SPANNDiskOptimized(
    dim=128,
    target_posting_size=200,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    clustering='hierarchical',  # Uses balanced k-means
    disk_path='./test_index_hierarchical'
)

print("Building index with hierarchical (balanced k-means)...")
start = time.time()
index2.build(base)
build_time2 = time.time() - start
print(f"Build time: {build_time2:.2f}s")

# Search
print("Searching...")
recalls2 = []
latencies2 = []

for i, query in enumerate(queries):
    start = time.time()
    dists, indices = index2.search(
        query, base, k=10,
        search_internal_result_num=64,
        max_check=4096
    )
    latency = (time.perf_counter() - start) * 1000
    latencies2.append(latency)
    
    # Calculate recall
    recall = len(set(indices) & set(ground_truth[i])) / 10
    recalls2.append(recall)

print(f"Recall@10: {np.mean(recalls2)*100:.2f}%")
print(f"Latency p50: {np.percentile(latencies2, 50):.2f}ms")
print(f"Latency p90: {np.percentile(latencies2, 90):.2f}ms")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Metric':<30} {'Standard':<15} {'Balanced':<15} {'Improvement'}")
print("-"*60)
print(f"{'Build Time (s)':<30} {build_time1:<15.2f} {build_time2:<15.2f} {(build_time1-build_time2)/build_time1*100:+.1f}%")
print(f"{'Recall@10 (%)':<30} {np.mean(recalls)*100:<15.2f} {np.mean(recalls2)*100:<15.2f} {(np.mean(recalls2)-np.mean(recalls))*100:+.2f}pp")
print(f"{'Latency p50 (ms)':<30} {np.percentile(latencies, 50):<15.2f} {np.percentile(latencies2, 50):<15.2f} {(np.percentile(latencies, 50)-np.percentile(latencies2, 50))/np.percentile(latencies, 50)*100:+.1f}%")
print(f"{'Latency p90 (ms)':<30} {np.percentile(latencies, 90):<15.2f} {np.percentile(latencies2, 90):<15.2f} {(np.percentile(latencies, 90)-np.percentile(latencies2, 90))/np.percentile(latencies, 90)*100:+.1f}%")

if np.mean(recalls2) > np.mean(recalls):
    print("\n✓ Balanced k-means improves recall!")
else:
    print("\n⚠ No significant recall improvement (may need larger dataset)")

print("\nCleaning up...")
import shutil
shutil.rmtree('./test_index_standard', ignore_errors=True)
shutil.rmtree('./test_index_hierarchical', ignore_errors=True)
print("Done!")
