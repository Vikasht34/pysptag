"""Quick local test on 10K"""
import numpy as np
import struct
import time
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

def read_fvecs(filename, max_vecs=None):
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

# Load 10K
base = read_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs', max_vecs=10000)
queries = read_fvecs('/Users/viktari/pysptag/data/sift/sift_query.fvecs', max_vecs=10)
groundtruth = read_ivecs('/Users/viktari/pysptag/data/sift/sift_groundtruth.ivecs')[:10]

print(f"Data: {base.shape}, Queries: {queries.shape}")

# TEST 1: No RaBitQ
print("\n" + "="*60)
print("TEST 1: No RaBitQ")
print("="*60)
index1 = SPANNRaBitQReplica(dim=128, target_posting_size=5000, replica_count=2, use_rabitq=False)
t0 = time.time()
index1.build(base)
print(f"Build: {time.time()-t0:.2f}s")

t0 = time.time()
recalls1 = []
for i, query in enumerate(queries):
    dists, indices = index1.search(query, base, k=10, max_check=2000)
    gt = set(groundtruth[i][:10])
    gt_in_10k = gt & set(range(10000))
    found = set(indices[:10])
    if len(gt_in_10k) > 0:
        recalls1.append(len(gt_in_10k & found) / len(gt_in_10k))
print(f"Search: {time.time()-t0:.2f}s")
print(f"Recall: {np.mean(recalls1):.2%}")

# TEST 2: With RaBitQ
print("\n" + "="*60)
print("TEST 2: With RaBitQ")
print("="*60)
index2 = SPANNRaBitQReplica(dim=128, target_posting_size=5000, replica_count=2, use_rabitq=True)
t0 = time.time()
index2.build(base)
print(f"Build: {time.time()-t0:.2f}s")

t0 = time.time()
recalls2 = []
for i, query in enumerate(queries):
    dists, indices = index2.search(query, base, k=10, max_check=2000)
    gt = set(groundtruth[i][:10])
    gt_in_10k = gt & set(range(10000))
    found = set(indices[:10])
    if len(gt_in_10k) > 0:
        recalls2.append(len(gt_in_10k & found) / len(gt_in_10k))
print(f"Search: {time.time()-t0:.2f}s")
print(f"Recall: {np.mean(recalls2):.2%}")

print("\n" + "="*60)
print(f"Recall drop: {(np.mean(recalls1) - np.mean(recalls2)):.2%}")
