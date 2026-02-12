"""Test SPANN+RaBitQ on SIFT1M dataset"""
import numpy as np
import struct
import time
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

def read_fvecs(filename, max_vecs=None):
    """Read .fvecs file format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
            if max_vecs and len(vectors) >= max_vecs:
                break
    return np.array(vectors, dtype=np.float32)

def read_ivecs(filename):
    """Read .ivecs file format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)

print("="*80)
print("SIFT1M Test - SPANN + RaBitQ + Replication")
print("="*80)

# Load SIFT1M
print("\nLoading SIFT1M dataset...")
t0 = time.time()
base = read_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs')
queries = read_fvecs('/Users/viktari/pysptag/data/sift/sift_query.fvecs')
groundtruth = read_ivecs('/Users/viktari/pysptag/data/sift/sift_groundtruth.ivecs')
load_time = time.time() - t0

print(f"✓ Loaded in {load_time:.2f}s")
print(f"  Base: {base.shape}")
print(f"  Queries: {queries.shape}")
print(f"  Groundtruth: {groundtruth.shape}")

# Build index
print("\nBuilding SPANN+RaBitQ index...")
print("  Parameters: replica_count=8, target_posting_size=10000, bq=4")
t0 = time.time()
index = SPANNRaBitQReplica(
    dim=128,
    target_posting_size=10000,
    replica_count=8,
    bq=4
)
index.build(base)
build_time = time.time() - t0

print(f"\n✓ Build time: {build_time:.2f}s ({build_time/60:.1f} min)")
print(f"  Clusters: {index.num_clusters}")
print(f"  Avg posting size: {np.mean([len(p) for p in index.posting_lists]):.1f}")

# Search
print(f"\nSearching {len(queries)} queries...")
t0 = time.time()
recalls_at_1 = []
recalls_at_10 = []
recalls_at_100 = []

for i in range(len(queries)):
    query = queries[i]
    dists, indices = index.search(
        query, base, k=100,
        search_internal_result_num=64,
        max_check=4096
    )
    
    # Compute recall
    gt = set(groundtruth[i][:100])
    found = set(indices)
    
    recalls_at_1.append(1 if indices[0] in gt else 0)
    recalls_at_10.append(len(found & set(groundtruth[i][:10])) / 10)
    recalls_at_100.append(len(found & gt) / 100)
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(queries)} queries...")

search_time = time.time() - t0
qps = len(queries) / search_time

print(f"\n✓ Search complete: {search_time:.2f}s")
print(f"  QPS: {qps:.2f}")
print(f"\n📊 Results:")
print(f"  Recall@1:   {np.mean(recalls_at_1):.2%}")
print(f"  Recall@10:  {np.mean(recalls_at_10):.2%}")
print(f"  Recall@100: {np.mean(recalls_at_100):.2%}")

print("\n" + "="*80)
print("✓ SIFT1M TEST COMPLETE")
print("="*80)
