"""
I/O Utilities - Dataset loading and file operations
"""
import h5py
import numpy as np
from typing import Tuple


def load_hdf5_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from HDF5 file
    
    Expected format:
        - train: (N, D) training vectors
        - test: (Q, D) query vectors
        - neighbors: (Q, K) ground truth indices
        - distances: (Q, K) ground truth distances (optional)
    
    Args:
        file_path: Path to HDF5 file
    
    Returns:
        train, test, neighbors
    """
    print(f"Loading dataset from {file_path}...")
    
    with h5py.File(file_path, 'r') as f:
        print(f"  Keys in file: {list(f.keys())}")
        
        train = f['train'][:]
        test = f['test'][:]
        neighbors = f['neighbors'][:]
        
        print(f"  Train: {train.shape}")
        print(f"  Test: {test.shape}")
        print(f"  Neighbors: {neighbors.shape}")
    
    return train, test, neighbors


def save_hdf5_dataset(
    file_path: str,
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray = None
):
    """
    Save dataset to HDF5 file
    
    Args:
        file_path: Path to save HDF5 file
        train: Training vectors
        test: Query vectors
        neighbors: Ground truth indices
        distances: Ground truth distances (optional)
    """
    print(f"Saving dataset to {file_path}...")
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('test', data=test)
        f.create_dataset('neighbors', data=neighbors)
        
        if distances is not None:
            f.create_dataset('distances', data=distances)
    
    print(f"✓ Saved")
