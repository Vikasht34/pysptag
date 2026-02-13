"""
Comprehensive metric test: Verify all metrics work correctly
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.index.spann_rabitq_replica import SPANNRaBitQReplica

np.random.seed(42)
n = 1000
dim = 128
data = np.random.randn(n, dim).astype(np.float32)
data = data / np.linalg.norm(data, axis=1, keepdims=True)  # Normalize

print("="*80)
print("Testing all metrics with quantization")
print("="*80)

for metric in ['L2', 'IP', 'Cosine']:
    print(f"\n{'='*80}")
    print(f"Metric: {metric}")
    print(f"{'='*80}")
    
    # Build
    index = SPANNRaBitQReplica(
        dim=dim,
        target_posting_size=100,
        replica_count=4,
        bq=1,
        use_rabitq=True,
        metric=metric
    )
    index.build(data)
    
    # Test search
    query = data[0]
    dists, indices = index.search(query, data, k=10)
    
    # Compute ground truth
    if metric == 'L2':
        gt_dists = np.sum((data - query) ** 2, axis=1)
    elif metric in ('IP', 'Cosine'):
        gt_dists = -np.dot(data, query)
    gt_indices = np.argsort(gt_dists)[:10]
    
    overlap = len(set(indices) & set(gt_indices))
    print(f"Returned: {indices[:5]}")
    print(f"Expected: {gt_indices[:5]}")
    print(f"Overlap: {overlap}/10")
    
    if 0 in indices:
        pos = np.where(indices == 0)[0][0]
        print(f"✓ Query found at position {pos}")
    else:
        print(f"✗ Query NOT found (BUG!)")
    
    if overlap >= 8:
        print(f"✓ {metric} metric PASSED")
    else:
        print(f"✗ {metric} metric FAILED")

print("\n" + "="*80)
print("Testing no-quantization mode")
print("="*80)

for metric in ['L2', 'IP', 'Cosine']:
    print(f"\n{metric} (no-quant): ", end="")
    
    index = SPANNRaBitQReplica(
        dim=dim,
        target_posting_size=100,
        replica_count=4,
        use_rabitq=False,
        metric=metric
    )
    index.build(data)
    
    query = data[0]
    dists, indices = index.search(query, data, k=10)
    
    if metric == 'L2':
        gt_dists = np.sum((data - query) ** 2, axis=1)
    else:
        gt_dists = -np.dot(data, query)
    gt_indices = np.argsort(gt_dists)[:10]
    
    overlap = len(set(indices) & set(gt_indices))
    if overlap >= 8 and 0 in indices:
        print(f"✓ PASSED ({overlap}/10)")
    else:
        print(f"✗ FAILED ({overlap}/10)")

print("\n" + "="*80)
print("✅ All metric tests complete!")
print("="*80)
