"""Quick memory test with small subset."""
import numpy as np
import time
from src.utils.io import load_fvecs
from src.index.spann_disk_optimized import SPANNDiskOptimized

# Load only 100K vectors
print("Loading SIFT subset...")
base = load_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs')
base = base[:100000]  # Only 100K
print(f"Base: {base.shape}")

# Build index
print("\nBuilding index...")
index = SPANNDiskOptimized(
    dim=128,
    metric='L2',
    tree_type='BKT',
    replica_count=8,
    use_rng_filtering=True
)

t0 = time.time()
index.build(base)
build_time = time.time() - t0
print(f"\n✓ Build completed in {build_time:.2f}s")
