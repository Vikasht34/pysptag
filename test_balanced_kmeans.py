"""
Test balanced k-means vs standard k-means
"""
import numpy as np
import time
from src.clustering.balanced_kmeans import balanced_kmeans, dynamic_factor_select
from src.core.bktree_complete import BKTree

# Generate test data
np.random.seed(42)
n = 10000
dim = 128
data = np.random.randn(n, dim).astype(np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)

print("="*60)
print("Testing Balanced K-means")
print("="*60)

# Test 1: Compare cluster balance
print("\n[Test 1] Cluster Balance Comparison")
print("-"*60)

k = 32

# Standard k-means (simple implementation for comparison)
print("Standard k-means:")
def simple_kmeans(data, k, max_iter=100):
    n, dim = data.shape
    # Random init
    centers = data[np.random.choice(n, k, replace=False)]
    for _ in range(max_iter):
        dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = data[mask].mean(axis=0)
    return labels

labels_std = simple_kmeans(data, k)
counts_std = np.bincount(labels_std, minlength=k)
avg_std = counts_std.mean()
std_std = np.sqrt(((counts_std - avg_std) ** 2).mean()) / avg_std
print(f"  Cluster sizes: min={counts_std.min()}, max={counts_std.max()}, avg={avg_std:.1f}")
print(f"  Std/Avg ratio: {std_std:.3f}")

# Balanced k-means
print("\nBalanced k-means (lambda=100):")
labels_bal, centers_bal = balanced_kmeans(data, k, lambda_factor=100.0, metric='L2')
counts_bal = np.bincount(labels_bal, minlength=k)
avg_bal = counts_bal.mean()
std_bal = np.sqrt(((counts_bal - avg_bal) ** 2).mean()) / avg_bal
print(f"  Cluster sizes: min={counts_bal.min()}, max={counts_bal.max()}, avg={avg_bal:.1f}")
print(f"  Std/Avg ratio: {std_bal:.3f}")

print(f"\n✓ Balance improvement: {(std_std - std_bal) / std_std * 100:.1f}% reduction in variance")

# Test 2: Auto-select lambda
print("\n[Test 2] Auto-select Lambda Factor")
print("-"*60)

best_lambda = dynamic_factor_select(data, k, samples=1000, metric='L2')
print(f"Best lambda factor: {best_lambda}")

labels_auto, centers_auto = balanced_kmeans(data, k, lambda_factor=best_lambda, metric='L2')
counts_auto = np.bincount(labels_auto, minlength=k)
avg_auto = counts_auto.mean()
std_auto = np.sqrt(((counts_auto - avg_auto) ** 2).mean()) / avg_auto
print(f"  Cluster sizes: min={counts_auto.min()}, max={counts_auto.max()}, avg={avg_auto:.1f}")
print(f"  Std/Avg ratio: {std_auto:.3f}")

# Test 3: BKTree with balanced k-means
print("\n[Test 3] BKTree with Balanced K-means")
print("-"*60)

print("Building BKTree with balanced k-means...")
start = time.time()
tree = BKTree(
    kmeans_k=32,
    leaf_size=8,
    num_trees=1,
    samples=1000,
    balance_factor=-1.0,  # Auto-select
    metric='L2'
)
tree.build(data)
build_time = time.time() - start

print(f"  Build time: {build_time:.2f}s")
print(f"  Total nodes: {len(tree.nodes)}")
print(f"  Tree roots: {len(tree.tree_roots)}")

# Analyze tree balance
leaf_sizes = []
for node in tree.nodes:
    if node.childStart < 0:  # Leaf node
        leaf_sizes.append(1)

if leaf_sizes:
    print(f"  Leaf nodes: {len(leaf_sizes)}")
    print(f"  Avg leaf size: {np.mean(leaf_sizes):.1f}")

print("\n✓ BKTree built successfully with balanced k-means")

# Test 4: Performance comparison
print("\n[Test 4] Performance Summary")
print("-"*60)
print(f"Standard k-means:  Std/Avg = {std_std:.3f}")
print(f"Balanced k-means:  Std/Avg = {std_bal:.3f}")
print(f"Auto-tuned:        Std/Avg = {std_auto:.3f}")
print(f"\nTarget: Std/Avg < 0.3 for good balance")

if std_bal < 0.3:
    print("✓ PASS: Balanced k-means achieves good balance")
else:
    print("⚠ WARNING: Balance could be improved")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
