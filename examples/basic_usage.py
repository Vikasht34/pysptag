"""
Basic Usage Example - PySPTAG

Demonstrates core functionality:
1. Build SPANN index
2. Search
3. Save/load from disk
4. Dynamic updates with SPFresh
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/viktari/pysptag/src')

from index.spann import SPANN
from storage.file_controller import SPANNDisk
from index.spfresh import SPFreshDynamic
import tempfile
from pathlib import Path

print("=" * 80)
print("PySPTAG - Basic Usage Example")
print("=" * 80)

# Generate sample data
np.random.seed(42)
dim = 128
n_train = 10000
n_test = 100
k = 10

print(f"\nDataset: {n_train:,} train, {n_test} test, dim={dim}")

train_vectors = np.random.randn(n_train, dim).astype('float32')
test_vectors = np.random.randn(n_test, dim).astype('float32')

# ============================================================================
# Example 1: In-Memory SPANN
# ============================================================================
print("\n" + "=" * 80)
print("Example 1: In-Memory SPANN")
print("=" * 80)

# Build index
print("\n[1] Building index...")
index = SPANN(
    dim=dim,
    target_posting_size=1000,
    closure_factor=1.5
)
index.build(train_vectors)

stats = index.get_stats()
print(f"  Clusters: {stats['n_clusters']}")
print(f"  Replication: {stats['replication_factor']:.2f}x")

# Search
print("\n[2] Searching...")
query = test_vectors[0]
distances, indices = index.search(query, k=k, n_probe=20)

print(f"  Top-{k} results:")
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f"    {i+1}. Vector {idx} (distance: {dist:.4f})")

# ============================================================================
# Example 2: Disk-Based SPANN
# ============================================================================
print("\n" + "=" * 80)
print("Example 2: Disk-Based SPANN")
print("=" * 80)

temp_dir = tempfile.mkdtemp()
index_dir = Path(temp_dir) / "index"

# Build and save
print("\n[1] Building and saving to disk...")
disk_index = SPANNDisk(
    dim=dim,
    index_dir=str(index_dir),
    target_posting_size=1000,
    closure_factor=1.5
)
disk_index.build(train_vectors)

print(f"  Index saved to: {index_dir}")
print(f"  Files: {list(index_dir.glob('**/*'))[:5]}")

# Load and search
print("\n[2] Loading from disk...")
disk_index2 = SPANNDisk(dim=dim, index_dir=str(index_dir))
disk_index2.load()

print("\n[3] Searching...")
distances, indices = disk_index2.search(query, train_vectors, k=k, n_probe=20)

print(f"  Top-{k} results:")
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f"    {i+1}. Vector {idx} (distance: {dist:.4f})")

# ============================================================================
# Example 3: SPFresh Dynamic Updates
# ============================================================================
print("\n" + "=" * 80)
print("Example 3: SPFresh Dynamic Updates")
print("=" * 80)

# Create SPFresh index
print("\n[1] Creating SPFresh index...")
spfresh = SPFreshDynamic(
    base_index=index,
    index_dir=str(index_dir),
    target_posting_size=1000
)

# Initialize posting sizes
for posting_id, posting in index.postings.items():
    spfresh.posting_sizes.update_size(posting_id, len(posting))

print(f"  Initialized with {len(index.postings)} postings")

# Insert new vectors
print("\n[2] Inserting new vectors...")
new_vectors = np.random.randn(100, dim).astype('float32')
vector_ids = spfresh.insert(new_vectors)

print(f"  Inserted {len(vector_ids)} vectors")
print(f"  Vector IDs: {vector_ids[:5]}...")

# Delete some vectors
print("\n[3] Deleting vectors...")
delete_ids = vector_ids[:10]
spfresh.delete(delete_ids)

print(f"  Deleted {len(delete_ids)} vectors")
print(f"  Active vectors: {spfresh.version_map.count_active()}")
print(f"  Deleted vectors: {spfresh.version_map.count_deleted()}")

# Check version tracking
print("\n[4] Version tracking...")
test_id = vector_ids[0]
print(f"  Vector {test_id}:")
print(f"    Version: {spfresh.version_map.get_version(test_id)}")
print(f"    Deleted: {spfresh.version_map.is_deleted(test_id)}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print("""
✅ In-Memory SPANN
   - Fast build and search
   - Good for small datasets
   - All data in RAM

✅ Disk-Based SPANN
   - Scalable to billions
   - Low memory footprint
   - EBS/SSD storage

✅ SPFresh Updates
   - Insert new vectors
   - Delete with tombstones
   - Automatic split/merge
   - Version tracking

Next: See examples/spfresh_updates.py for advanced SPFresh usage
""")

# Cleanup
import shutil
shutil.rmtree(temp_dir)

print("=" * 80)
