"""
Test RNG filtering impact on recall
"""
import numpy as np
import time
from src.index.spann_disk_optimized import SPANNDiskOptimized
from src.utils.io import load_fvecs

print("="*60)
print("RNG Filtering Impact Test")
print("="*60)

# Load SIFT data
print("\nLoading SIFT dataset...")
base = load_fvecs('data/sift/sift_base.fvecs')[:50000]  # 50K for faster testing
queries = load_fvecs('data/sift/sift_query.fvecs')[:100]
ground_truth = np.fromfile('data/sift/sift_groundtruth.ivecs', dtype=np.int32)
ground_truth = ground_truth.reshape(-1, 101)[:100, 1:11]

print(f"Base: {base.shape}")
print(f"Queries: {queries.shape}")

# Test 1: Without RNG filtering
print("\n[Test 1] Without RNG Filtering (Simple Top-K)")
print("-"*60)

index1 = SPANNDiskOptimized(
    dim=128,
    target_posting_size=200,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    clustering='kmeans',
    use_rng_filtering=False,  # Disable RNG
    disk_path='./test_index_no_rng'
)

print("Building index...")
start = time.time()
index1.build(base)
build_time1 = time.time() - start
print(f"Build time: {build_time1:.2f}s")

# Search
print("Searching...")
recalls1 = []
latencies1 = []

for i, query in enumerate(queries):
    start = time.perf_counter()
    dists, indices = index1.search(
        query, base, k=10,
        search_internal_result_num=64,
        max_check=4096
    )
    latency = (time.perf_counter() - start) * 1000
    latencies1.append(latency)
    
    recall = len(set(indices) & set(ground_truth[i])) / 10
    recalls1.append(recall)

print(f"Recall@10: {np.mean(recalls1)*100:.2f}%")
print(f"Latency p50: {np.percentile(latencies1, 50):.2f}ms")
print(f"Latency p90: {np.percentile(latencies1, 90):.2f}ms")

# Test 2: With RNG filtering
print("\n[Test 2] With RNG Filtering (SPTAG NPA)")
print("-"*60)

index2 = SPANNDiskOptimized(
    dim=128,
    target_posting_size=200,
    replica_count=8,
    use_rabitq=False,
    metric='L2',
    clustering='kmeans',
    use_rng_filtering=True,  # Enable RNG
    disk_path='./test_index_with_rng'
)

print("Building index...")
start = time.time()
index2.build(base)
build_time2 = time.time() - start
print(f"Build time: {build_time2:.2f}s")

# Search
print("Searching...")
recalls2 = []
latencies2 = []

for i, query in enumerate(queries):
    start = time.perf_counter()
    dists, indices = index2.search(
        query, base, k=10,
        search_internal_result_num=64,
        max_check=4096
    )
    latency = (time.perf_counter() - start) * 1000
    latencies2.append(latency)
    
    recall = len(set(indices) & set(ground_truth[i])) / 10
    recalls2.append(recall)

print(f"Recall@10: {np.mean(recalls2)*100:.2f}%")
print(f"Latency p50: {np.percentile(latencies2, 50):.2f}ms")
print(f"Latency p90: {np.percentile(latencies2, 90):.2f}ms")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Metric':<30} {'No RNG':<15} {'With RNG':<15} {'Improvement'}")
print("-"*60)
print(f"{'Build Time (s)':<30} {build_time1:<15.2f} {build_time2:<15.2f} {(build_time2-build_time1)/build_time1*100:+.1f}%")
print(f"{'Recall@10 (%)':<30} {np.mean(recalls1)*100:<15.2f} {np.mean(recalls2)*100:<15.2f} {(np.mean(recalls2)-np.mean(recalls1))*100:+.2f}pp")
print(f"{'Latency p50 (ms)':<30} {np.percentile(latencies1, 50):<15.2f} {np.percentile(latencies2, 50):<15.2f} {(np.percentile(latencies2, 50)-np.percentile(latencies1, 50))/np.percentile(latencies1, 50)*100:+.1f}%")
print(f"{'Latency p90 (ms)':<30} {np.percentile(latencies1, 90):<15.2f} {np.percentile(latencies2, 90):<15.2f} {(np.percentile(latencies2, 90)-np.percentile(latencies1, 90))/np.percentile(latencies1, 90)*100:+.1f}%")

recall_improvement = (np.mean(recalls2) - np.mean(recalls1)) * 100

if recall_improvement > 2:
    print(f"\n✓ SUCCESS: RNG filtering improves recall by {recall_improvement:.1f} percentage points!")
elif recall_improvement > 0:
    print(f"\n⚠ MINOR: RNG filtering improves recall by {recall_improvement:.1f}pp (expected 5-10pp)")
else:
    print(f"\n✗ FAIL: No recall improvement (check implementation)")

print("\nCleaning up...")
import shutil
shutil.rmtree('./test_index_no_rng', ignore_errors=True)
shutil.rmtree('./test_index_with_rng', ignore_errors=True)
print("Done!")
