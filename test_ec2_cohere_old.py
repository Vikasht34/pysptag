"""
EC2 Test Script for SPTAG-Optimized Disk-Based SPANN on Cohere 1M
Uses ratio-based clustering (1% of data as centroids)

Dataset: Cohere 1M (768-dim embeddings)
Format: HDF5

Target: p90 <10ms with 90%+ recall on EBS

Prerequisites:
- EC2 instance with sufficient RAM (32GB+)
- EBS volume mounted (for disk storage)
- Cohere 1M dataset downloaded

Usage:
    python3 test_ec2_cohere.py
"""
import numpy as np
import h5py
import time
import sys
import os
sys.path.insert(0, '.')
from src.index.spann_disk_optimized import SPANNDiskOptimized

def get_dir_size(path):
    """Get directory size in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

print("="*80)
print("EC2: SPTAG-Optimized Disk-Based SPANN on Cohere 1M")
print("="*80)

# Configuration
DATA_FILE = os.path.expanduser('~/pysptag/cohere-wikipedia-768-angular.hdf5')
INDEX_DIR = '/mnt/ebs/spann_cohere_sptag'  # EBS mount point
NUM_QUERIES = 100

# Load Cohere 1M
print("\nLoading Cohere 1M dataset...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")
print(f"✓ Dimension: {base.shape[1]}")

# Build SPTAG-optimized index ONCE (clustering is expensive)
print("\n" + "="*80)
print("Building SPTAG-Optimized Index (clustering only)")
print("="*80)
print("  Config: ratio=0.01 (10K clusters), replica=8")
print("  Posting limit: 500 (adaptive for 1M)")

disk_path = INDEX_DIR

print(f"\nBuilding index...")
t0 = time.time()
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=False,  # Will test different quantizations
    metric='Cosine',
    tree_type='KDT',
    disk_path=disk_path,
    cache_size=1024,
    clustering='kmeans'
)
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")

disk_size = get_dir_size(disk_path)
print(f"✓ Disk usage: {disk_size/1024**3:.2f} GB")
print(f"✓ Clusters: {index.num_clusters}")

# Test different quantization levels
configs = [
    ('no-quant', False, 4),
    ('1-bit', True, 1),
    ('2-bit', True, 2),
    ('4-bit', True, 4),
]

results = []

for name, use_rabitq, bq in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print("="*80)
    
    # Update quantization settings
    index.use_rabitq = use_rabitq
    index.bq = bq
    if use_rabitq:
        from src.quantization.rabitq_numba import RaBitQNumba
        index.rabitq = RaBitQNumba(dim=768, bq=bq)
        index.rabitq.train(base[:10000])  # Train on subset
    
    # Warm up
    print("Warming up cache...")
    for q in queries[:20]:
        index.search(q, base, k=10, search_internal_result_num=64, max_check=8192)
    
    # Test
    print(f"Testing ({NUM_QUERIES} queries)...")
    t0 = time.time()
    recalls = []
    latencies = []
    latencies = []
    
    for i, query in enumerate(queries):
        q_start = time.time()
for i, query in enumerate(queries):
    q_start = time.time()
    dists, indices = index.search(
        query, base, k=10, 
        search_internal_result_num=64,
        max_check=8192
    )
    latencies.append((time.time() - q_start) * 1000)
    
    if len(indices) == 0:
        recalls.append(0)
        continue
    
    gt = set(groundtruth[i][:10])
    found = set(indices[:10])
    recalls.append(len(gt & found) / 10)
    
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{NUM_QUERIES} queries...")

search_time = time.time() - t0
avg_recall = np.mean(recalls) * 100
p50 = np.percentile(latencies, 50)
p90 = np.percentile(latencies, 90)
p99 = np.percentile(latencies, 99)
qps = NUM_QUERIES / search_time

print(f"\n{'='*80}")
print("RESULTS")
print("="*80)
print(f"Latency:")
print(f"  p50: {p50:.2f}ms")
print(f"  p90: {p90:.2f}ms {'✅' if p90 < 10 else '❌'} (target <10ms)")
print(f"  p99: {p99:.2f}ms")
print(f"\nRecall@10: {avg_recall:.1f}% {'✅' if avg_recall >= 90 else '❌'} (target ≥90%)")
print(f"\nPerformance:")
print(f"  QPS: {qps:.1f}")
print(f"  Search time: {search_time:.1f}s")
print(f"  Build time: {build_time:.1f}s ({build_time/60:.1f} min)")
print(f"\nIndex stats:")
print(f"  Clusters: {index.num_clusters}")
print(f"  Disk usage: {disk_size/1024**3:.2f} GB")
print(f"  Cache hit rate: {index._cache_hits/(index._cache_hits+index._cache_misses)*100:.1f}%")

if p90 < 10 and avg_recall >= 90:
    print(f"\n✅ SUCCESS: p90={p90:.2f}ms with {avg_recall:.1f}% recall on EBS!")
else:
    print(f"\n⚠️  Needs tuning:")
    if p90 >= 10:
        print(f"   - p90 too high: {p90:.2f}ms (target <10ms)")
if p90 < 10 and avg_recall >= 90:
    print(f"\n✅ SUCCESS: p90={p90:.2f}ms with {avg_recall:.1f}% recall on EBS!")
else:
    print(f"\n⚠️  Needs tuning:")
    if p90 >= 10:
        print(f"   - p90 too high: {p90:.2f}ms (target <10ms)")
    if avg_recall < 90:
        print(f"   - Recall too low: {avg_recall:.1f}% (target ≥90%)")

print(f"\n{'='*80}")
print(f"To cleanup: rm -rf {INDEX_DIR}")
print("="*80)
