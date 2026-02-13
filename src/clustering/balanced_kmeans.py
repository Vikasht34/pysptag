"""
Balanced K-means with lambda penalty (SPTAG-style)
Ensures clusters are roughly equal size
"""
import numpy as np
from typing import Tuple


def balanced_kmeans(
    data: np.ndarray,
    k: int,
    lambda_factor: float = 100.0,
    max_iter: int = 100,
    metric: str = 'L2'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced k-means with penalty term to prevent imbalanced clusters.
    
    Args:
        data: (N, D) array
        k: Number of clusters
        lambda_factor: Initial penalty factor
        max_iter: Max iterations
        metric: Distance metric
        
    Returns:
        labels: (N,) cluster assignments
        centers: (K, D) cluster centers
    """
    n, dim = data.shape
    
    # Initialize centers (k-means++)
    centers = _init_centers_kmeans_pp(data, k, metric)
    counts = np.zeros(k, dtype=np.int32)
    
    for iteration in range(max_iter):
        # Compute distances
        if metric == 'L2':
            dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        elif metric in ('IP', 'Cosine'):
            dists = -np.dot(data, centers.T)
        else:
            dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        
        # Add penalty term (SPTAG's key innovation)
        penalty = lambda_factor * counts[None, :]
        adjusted_dists = dists + penalty
        
        # Assign to nearest (with penalty)
        labels = np.argmin(adjusted_dists, axis=1)
        
        # Update counts
        new_counts = np.bincount(labels, minlength=k)
        
        # Refine lambda (SPTAG's adaptive lambda)
        if new_counts.max() > 0:
            max_cluster = np.argmax(new_counts)
            mask = labels == max_cluster
            if mask.any():
                cluster_dists = dists[mask, max_cluster]
                avg_dist = np.mean(cluster_dists)
                max_dist = np.max(cluster_dists)
                lambda_factor = max(0, (max_dist - avg_dist) / n)
        
        counts = new_counts
        
        # Update centers
        old_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = data[mask].mean(axis=0)
                if metric == 'Cosine':
                    norm = np.linalg.norm(centers[i])
                    if norm > 0:
                        centers[i] /= norm
        
        # Check convergence
        diff = np.sum((centers - old_centers) ** 2)
        if diff < 1e-3:
            break
    
    return labels, centers


def _init_centers_kmeans_pp(data: np.ndarray, k: int, metric: str) -> np.ndarray:
    """K-means++ initialization"""
    n, dim = data.shape
    centers = np.zeros((k, dim), dtype=data.dtype)
    
    # First center: random
    centers[0] = data[np.random.randint(n)]
    
    # Remaining centers: weighted by distance
    for i in range(1, k):
        if metric == 'L2':
            dists = np.min(np.sum((data[:, None, :] - centers[None, :i, :]) ** 2, axis=2), axis=1)
        elif metric in ('IP', 'Cosine'):
            dists = -np.max(np.dot(data, centers[:i].T), axis=1)
            dists = dists - dists.min() + 1e-8  # Shift to positive
        else:
            dists = np.min(np.sum((data[:, None, :] - centers[None, :i, :]) ** 2, axis=2), axis=1)
        
        probs = dists / dists.sum()
        centers[i] = data[np.random.choice(n, p=probs)]
    
    return centers


def dynamic_factor_select(
    data: np.ndarray,
    k: int,
    samples: int = 1000,
    metric: str = 'L2'
) -> float:
    """
    Auto-select best lambda factor (SPTAG's DynamicFactorSelect).
    
    Args:
        data: (N, D) array
        k: Number of clusters
        samples: Sample size for testing
        metric: Distance metric
        
    Returns:
        Best lambda factor
    """
    n = len(data)
    
    # Sample data if too large
    if samples > 0 and samples < n:
        indices = np.random.choice(n, samples, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data
    
    best_lambda = 100.0
    best_std = float('inf')
    
    # Try different lambda factors
    for lambda_factor in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        labels, _ = balanced_kmeans(sample_data, k, lambda_factor, max_iter=20, metric=metric)
        
        # Calculate cluster size standard deviation
        counts = np.bincount(labels, minlength=k)
        if counts.sum() > 0:
            avg = counts.mean()
            std = np.sqrt(((counts - avg) ** 2).mean()) / avg if avg > 0 else float('inf')
            
            if std < best_std:
                best_std = std
                best_lambda = lambda_factor
    
    return best_lambda
