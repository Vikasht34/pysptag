"""SPTAG's lambda-balanced k-means implementation."""
import numpy as np
from typing import Tuple


def lambda_balanced_kmeans(
    data: np.ndarray,
    k: int,
    samples: int = 1000,
    lambda_factor: float = 100.0,
    max_iter: int = 100,
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SPTAG's lambda-balanced k-means clustering.
    
    Args:
        data: (n, d) array of vectors
        k: Number of clusters
        samples: Number of samples for initialization and iteration
        lambda_factor: Balance factor (higher = more balanced)
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        labels: (n,) cluster assignments
        centers: (k, d) cluster centers
    """
    np.random.seed(random_state)
    n, d = data.shape
    
    # Initialize centers from random samples
    sample_indices = np.random.choice(n, min(samples, n), replace=False)
    center_indices = np.random.choice(sample_indices, k, replace=False)
    centers = data[center_indices].copy()
    
    # Compute lambda
    base = np.mean(np.abs(data))
    lambda_val = base * base / lambda_factor / min(samples, n)
    
    # Batch size for iterations
    batch_size = min(samples, n)
    
    counts = np.zeros(k, dtype=np.int32)
    labels = np.zeros(n, dtype=np.int32)
    
    min_dist = float('inf')
    no_improvement = 0
    
    for iteration in range(max_iter):
        # Shuffle and batch
        indices = np.random.permutation(n)
        batch_indices = indices[:batch_size]
        batch_data = data[batch_indices]
        
        # Vectorized assignment with lambda balancing
        dists = np.sum((batch_data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        dists += lambda_val * counts[np.newaxis, :]
        labels_batch = np.argmin(dists, axis=1)
        
        # Update counts and centers
        new_counts = np.bincount(labels_batch, minlength=k)
        new_centers = np.zeros((k, d), dtype=np.float32)
        for k_idx in range(k):
            mask = labels_batch == k_idx
            if np.any(mask):
                new_centers[k_idx] = np.sum(batch_data[mask], axis=0)
        
        total_dist = np.sum(np.min(dists, axis=1))
        counts = new_counts.copy()
        
        # Check convergence
        if total_dist < min_dist:
            no_improvement = 0
            min_dist = total_dist
        else:
            no_improvement += 1
        
        # Update centers
        center_diff = 0.0
        for k_idx in range(k):
            if new_counts[k_idx] > 0:
                new_center = new_centers[k_idx] / new_counts[k_idx]
                center_diff += np.sum((centers[k_idx] - new_center) ** 2)
                centers[k_idx] = new_center
        
        if center_diff < 1e-3 or no_improvement >= 5:
            break
    
    # Final assignment on full dataset without lambda (vectorized)
    dists = np.sum((data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    counts = np.bincount(labels, minlength=k)
    
    # Replace centers with actual data points (SPTAG does this)
    for k_idx in range(k):
        if counts[k_idx] > 0:
            cluster_points = np.where(labels == k_idx)[0]
            # Use closest point to center as the actual center
            dists = np.sum((data[cluster_points] - centers[k_idx]) ** 2, axis=1)
            centers[k_idx] = data[cluster_points[np.argmin(dists)]]
    
    return labels, centers
