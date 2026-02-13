"""
Base clustering interface
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List


class ClusteringAlgorithm(ABC):
    """Base class for clustering algorithms"""
    
    @abstractmethod
    def cluster(
        self,
        data: np.ndarray,
        target_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster data into target_clusters groups.
        
        Args:
            data: (N, D) array of vectors
            target_clusters: Target number of clusters
            
        Returns:
            centroids: (K, D) array of cluster centers
            labels: (N,) array of cluster assignments
        """
        pass
    
    @abstractmethod
    def assign_with_replicas(
        self,
        data: np.ndarray,
        centroids: np.ndarray,
        replica_count: int,
        posting_limit: int
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Assign vectors to multiple nearest centroids with posting limits.
        
        Args:
            data: (N, D) array of vectors
            centroids: (K, D) array of cluster centers
            replica_count: Number of replicas per vector
            posting_limit: Max vectors per posting
            
        Returns:
            postings: List of vector indices per centroid
            replica_counts: Number of replicas per vector
        """
        pass
