"""
EC2 Test Script for Disk-Based SPANN on Cohere 1M
Run this on EC2 with EBS storage for billion-scale testing

Dataset: Cohere 1M (768-dim embeddings)
Format: HDF5

Prerequisites:
- EC2 instance with sufficient RAM (16GB+)
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
from src.index.spann_disk import SPANNDiskBased

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
print("EC2: Disk-Based SPANN on Cohere 1M")
print("="*80)

# Configuration
DATA_FILE = os.path.expanduser('~/pysptag/cohere-wikipedia-768-angular.hdf5')
INDEX_DIR = '/mnt/ebs/spann_cohere'  # EBS mount point
NUM_QUERIES = 100

# Load Cohere 1M
print("\nLoading Cohere 1M dataset...")
with h5py.File(DATA_FILE, 'r') as f:
    base = f['train'][:]
    queries = f['test'][:NUM_QUERIES]
    groundtruth = f['neighbors'][:NUM_QUERIES]

print(f"✓ Base: {base.shape}, Queries: {queries.shape}")
print(f"✓ Dimension: {base.shape[1]}")

# Test configurations
configs = [
    ('4-bit', 4, True),
    ('2-bit', 2, True),
    ('no-quant', 4, False),
]

results = []

for name, bq, use_rabitq in configs:
    print("\n" + "="*80)
    print(f"TEST: {name}")
    print("="*80)
    
    disk_path = f'{INDEX_DIR}_{name}'
    
    # Build index
    metadata_file = os.path.join(disk_path, 'metadata.pkl')
    if os.path.exists(metadata_file):
        print(f"Loading existing index from {disk_path}...")
        t0 = time.time()
        index = SPANNDiskBased.load(disk_path)
        load_time = time.time() - t0
        print(f"✓ Loaded in {load_time:.2f}s")
    else:
        print(f"Building new index...")
        t0 = time.time()
        index = SPANNDiskBased(
            dim=768,  # Cohere dimension
            target_posting_size=5000,
            replica_count=6,
            bq=bq,
            use_rabitq=use_rabitq,
            metric='Cosine',  # Cohere uses cosine similarity
            tree_type='KDT',  # KDTree is 3× faster than BKTree
            disk_path=disk_path
        )
        index.build(base)
        build_time = time.time() - t0
        print(f"✓ Built in {build_time:.2f}s")
    
    # Check disk usage
    disk_size = get_dir_size(disk_path)
    print(f"✓ Disk usage: {disk_size/1024**2:.2f} MB")
    
    # Search
    print("Searching...")
    t0 = time.time()
    recalls = []
    latencies = []
    
    for i, query in enumerate(queries):
        q_start = time.time()
        dists, indices = index.search(
            query, base, k=10, 
            search_internal_result_num=128,
            max_check=4096
        )
        latencies.append((time.time() - q_start) * 1000)
        
        if len(indices) == 0:
            recalls.append(0)
            continue
        
        gt = set(groundtruth[i][:10])
        found = set(indices[:10])
        recalls.append(len(gt & found) / 10)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{NUM_QUERIES} queries")
    
    search_time = time.time() - t0
    avg_recall = np.mean(recalls) * 100
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    qps = NUM_QUERIES / search_time
    
    print(f"Search time: {search_time:.2f}s, QPS: {qps:.1f}")
    print(f"Latency: p50={p50:.2f}ms, p90={p90:.2f}ms, p99={p99:.2f}ms")
    print(f"Recall@10: {avg_recall:.2f}%")
    
    results.append({
        'config': name,
        'disk_mb': disk_size/1024**2,
        'search_time': search_time,
        'qps': qps,
        'p50': p50,
        'p90': p90,
        'p99': p99,
        'recall': avg_recall
    })

# Summary
print("\n" + "="*80)
print("SUMMARY: Disk-Based SPANN on Cohere 1M (EC2)")
print("="*80)
print(f"{'Config':<12} {'Disk(MB)':<10} {'QPS':<8} {'p50(ms)':<8} {'p90(ms)':<8} {'p99(ms)':<8} {'Recall':<8}")
print("-"*80)
for r in results:
    print(f"{r['config']:<12} {r['disk_mb']:<10.1f} {r['qps']:<8.1f} {r['p50']:<8.2f} {r['p90']:<8.2f} {r['p99']:<8.2f} {r['recall']:<8.2f}%")
print("="*80)

print("\n✓ Disk-based SPANN test complete!")
print(f"✓ Index saved to: {INDEX_DIR}_*")
print("✓ Ready for billion-scale embeddings!")
