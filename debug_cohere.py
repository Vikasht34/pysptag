"""
Debug: Check Cohere 1M dataset
"""
import h5py
import numpy as np

data_file = '/Users/viktari/pysptag/data/documents-1m.hdf5'

with h5py.File(data_file, 'r') as f:
    print("Keys in HDF5:", list(f.keys()))
    print()
    
    for key in f.keys():
        print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
        if len(f[key].shape) == 2:
            sample = f[key][0]
            print(f"  Sample norm: {np.linalg.norm(sample):.4f}")
            print(f"  Sample range: [{sample.min():.4f}, {sample.max():.4f}]")
        print()
    
    # Check if vectors are normalized
    train = f['train'][:1000]
    norms = np.linalg.norm(train, axis=1)
    print(f"Train vector norms (first 1000):")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Min: {norms.min():.4f}")
    print(f"  Max: {norms.max():.4f}")
    print(f"  Are normalized? {np.allclose(norms, 1.0)}")
