"""Test different bit quantizations: 1-bit, 2-bit, 4-bit"""
import numpy as np
import sys
sys.path.insert(0, '/Users/viktari/pysptag')

from src.quantization.rabitq import RaBitQ

np.random.seed(42)
data = np.random.randn(1000, 128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

# True distances
true_dists = np.sum((data - query) ** 2, axis=1)
true_top10 = set(np.argsort(true_dists)[:10])
true_top100 = set(np.argsort(true_dists)[:100])

print("="*70)
print("RaBitQ Quantization Comparison")
print("="*70)
print(f"Dataset: {data.shape[0]} vectors, {data.shape[1]} dimensions")
print(f"Testing recall@10 and recall@100\n")

for bq in [1, 2, 4]:
    print(f"{'='*70}")
    print(f"{bq}-bit Quantization")
    print(f"{'='*70}")
    
    rabitq = RaBitQ(dim=128, bq=bq)
    codes = rabitq.build(data)
    
    # Test recall@10
    est_dists10, est_indices10 = rabitq.search(query, codes, None, k=10)
    overlap10 = len(true_top10 & set(est_indices10))
    
    # Test recall@100
    est_dists100, est_indices100 = rabitq.search(query, codes, None, k=100)
    overlap100 = len(true_top100 & set(est_indices100))
    
    # Memory usage
    memory_mb = codes.nbytes / 1024 / 1024
    compression = data.nbytes / codes.nbytes
    
    print(f"  Recall@10:     {overlap10}/10 ({overlap10*10}%)")
    print(f"  Recall@100:    {overlap100}/100 ({overlap100}%)")
    print(f"  Memory:        {memory_mb:.2f} MB")
    print(f"  Compression:   {compression:.1f}x")
    print(f"  Levels:        {2**bq}")
    print()

print("="*70)
print("Summary:")
print("  1-bit: Highest compression, lowest recall")
print("  2-bit: Balanced")
print("  4-bit: Best recall, still good compression")
print("="*70)
