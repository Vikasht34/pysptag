"""
Test with SPTAG exact configuration:
- posting_size=118
- replica=8
- NO quantization (like SPTAG default)
- search_centroids=64
- max_check=4096
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')
from profile_latency import ProfiledSPANN

print("="*80)
print("SPTAG Exact Configuration (No Quantization)")
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

print("\nBuilding index (SPTAG exact config)...")
print("  posting_size=118")
print("  replica=8")
print("  use_rabitq=FALSE (no quantization, like SPTAG)")
print("  search_centroids=64")
print("  max_check=4096")

index = ProfiledSPANN(
    dim=768,
    target_posting_size=118,
    replica_count=8,
    bq=4,
    use_rabitq=False,  # NO QUANTIZATION like SPTAG!
    tree_type='KDT',
    disk_path='./sptag_exact',
    cache_size=256
)

t0 = time.time()
index.build(data)
build_time = time.time() - t0

print(f"✓ Built: {index.num_clusters} clusters in {build_time:.2f}s")

# Warm up
for query in queries[:10]:
    index.search(query, data, k=10, search_internal_result_num=64, max_check=4096)

# Test
print("\nTesting (100 queries)...")
index.reset_stats()
recalls = []
latencies = []

for i, query in enumerate(queries):
    t0 = time.time()
    dists, indices = index.search(query, data, k=10, 
                                  search_internal_result_num=64,
                                  max_check=4096)
    latencies.append((time.time() - t0) * 1000)
    
    gt = set(gt_indices[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)

avg_recall = np.mean(recalls) * 100
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p99 = np.percentile(latencies, 99)

print(f"\n{'='*80}")
print("RESULTS (No Quantization)")
print("="*80)
print(f"Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
print(f"Recall@10: {avg_recall:.1f}%")
print(f"QPS: {1000/p50:.1f}")

index.print_detailed_stats()

# Compare with quantized
print(f"\n{'='*80}")
print("COMPARISON: No-Quant vs 4-bit Quantization")
print("="*80)
print("No-Quant (this run):")
print(f"  Latency: {p50:.2f}ms")
print(f"  Recall: {avg_recall:.1f}%")
print(f"  Disk: {np.mean(index.stats['total_data_loaded_mb']):.2f} MB/query")
print("\n4-bit Quantized (previous):")
print(f"  Latency: 5.43ms")
print(f"  Recall: 65.9%")
print(f"  Disk: 58.83 MB/query")

if avg_recall > 85 and p50 < 10:
    print(f"\n✅ SUCCESS: {p50:.2f}ms with {avg_recall:.1f}% recall!")
elif avg_recall > 85:
    print(f"\n⚠️  Good recall ({avg_recall:.1f}%) but slow ({p50:.2f}ms)")
elif p50 < 10:
    print(f"\n⚠️  Fast ({p50:.2f}ms) but low recall ({avg_recall:.1f}%)")
else:
    print(f"\n❌ Need improvement: {p50:.2f}ms, {avg_recall:.1f}% recall")

# Cleanup
import shutil
shutil.rmtree('./sptag_exact', ignore_errors=True)
