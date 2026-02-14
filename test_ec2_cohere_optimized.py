"""
EC2 Test Script - SPTAG SPANN with ALL Optimizations on Cohere 1M
Target: <10ms p50 with 90%+ recall on EC2 NVMe

Optimizations:
1. Single-file format (2.4× speedup)
2. Faiss centroid search (40× faster)
3. RaBitQ 2-bit quantization
4. Shared RaBitQ instance (no JIT overhead)
5. Hierarchical balanced clustering
6. RNG graph on centroids
7. Batch loading
"""
import numpy as np
import h5py
import time
import sys
import os
import faiss
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
print("EC2: SPTAG SPANN with ALL Optimizations - Cohere 1M")
print("="*80)

# Configuration
DATA_FILE = os.path.expanduser('~/pysptag/cohere-wikipedia-768-angular.hdf5')
INDEX_DIR = '/mnt/nvme/spann_cohere_optimized'  # Use NVMe for best performance
NUM_QUERIES = 1000  # Test on more queries

# Load Cohere 1M
print("\nLoading Cohere 1M...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")

# Build index with ALL optimizations
print("\n" + "="*80)
print("Building Optimized Index")
print("="*80)
print("  Clustering: hierarchical (SPTAG-style)")
print("  Posting size: 500 (balanced)")
print("  Replica: 8 (SPTAG default)")
print("  Quantization: RaBitQ 2-bit")
print("  Centroid search: Faiss")
print("  Format: Single-file")

print(f"\nBuilding...")
t0 = time.time()
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=True,
    bq=2,  # 2-bit quantization
    metric='IP',
    tree_type='BKT',  # BKTree for better accuracy
    clustering='hierarchical',  # SPTAG-style
    use_rng_filtering=True,
    use_faiss_centroids=True,  # Fast centroid search
    disk_path=INDEX_DIR,
    cache_size=5000  # Large cache for EC2
)
index.build(base)
build_time = time.time() - t0
print(f"✓ Built in {build_time:.1f}s ({build_time/60:.1f} min)")

disk_size = get_dir_size(INDEX_DIR)
print(f"✓ Clusters: {index.num_clusters}")
print(f"✓ Disk: {disk_size/1024**3:.2f} GB")
print(f"✓ Single file: {os.path.exists(os.path.join(INDEX_DIR, 'postings.bin'))}")

# Warm up
print("\nWarming up (20 queries)...")
for q in queries[:20]:
    index.search(q, base, k=10, search_internal_result_num=48, max_check=6144)

# Test with different search parameters
configs = [
    ('centroids=32', 32),
    ('centroids=48', 48),
    ('centroids=64', 64),
]

results = []

for name, num_centroids in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print("="*80)
    
    latencies = []
    recalls = []
    
    for i, query in enumerate(queries):
        q_start = time.perf_counter()
        indices, dists = index.search(
            query, base, k=10, 
            search_internal_result_num=num_centroids,
            max_check=6144,
            use_async_pruning=True  # Enable SPTAG-style async pruning
        )
        latencies.append((time.perf_counter() - q_start) * 1000)
        
        if len(indices) == 0:
            recalls.append(0)
            continue
        
        gt = set(groundtruth[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{NUM_QUERIES}...")
    
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    print(f"\nResults:")
    print(f"  Recall:  {avg_recall:.2f}% {'✅' if avg_recall >= 90 else '⚠️'}")
    print(f"  p50:     {p50:.2f}ms {'✅' if p50 < 10 else '⚠️'}")
    print(f"  p90:     {p90:.2f}ms")
    print(f"  p95:     {p95:.2f}ms")
    print(f"  p99:     {p99:.2f}ms")
    print(f"  avg:     {avg:.2f}ms")
    
    results.append({
        'config': name,
        'p50': p50,
        'p90': p90,
        'p95': p95,
        'p99': p99,
        'avg': avg,
        'recall': avg_recall
    })

# Summary
print(f"\n{'='*80}")
print("FINAL SUMMARY: Cohere 1M on EC2")
print("="*80)
print(f"Build: {build_time:.1f}s, Clusters: {index.num_clusters}, Disk: {disk_size/1024**3:.2f} GB")
print()
print(f"{'Config':<15} {'p50(ms)':<10} {'p90(ms)':<10} {'Recall%':<10} {'Status':<15}")
print("-"*80)
for r in results:
    status = ''
    if r['p50'] < 10 and r['recall'] >= 90:
        status = '🎉 BOTH TARGETS'
    elif r['recall'] >= 90:
        status = '✅ Recall'
    elif r['p50'] < 10:
        status = '✅ Latency'
    
    print(f"{r['config']:<15} {r['p50']:<10.2f} {r['p90']:<10.2f} {r['recall']:<10.1f} {status:<15}")

print("\nOptimizations applied:")
print("  ✅ Single-file format")
print("  ✅ Faiss centroid search")
print("  ✅ RaBitQ 2-bit quantization")
print("  ✅ Hierarchical clustering")
print("  ✅ RNG graph")
print("  ✅ Batch loading")
print("  ✅ Large cache (5000)")
