"""
SPTAG-style dynamic head selection for clustering.
Replicates SPTAG's SelectHeadDynamically algorithm.
"""
import numpy as np
from typing import List, Tuple
import sys
sys.path.insert(0, '.')
from src.core.bktree import BKTree


class SPTAGClustering:
    """SPTAG-style dynamic clustering"""
    
    def __init__(
        self,
        select_threshold: int = 6,
        split_threshold: int = 25,
        split_factor: int = 2,
        ratio: float = 0.01,  # 1% of vectors as centroids
        posting_vector_limit: int = 118,
        replica_count: int = 8
    ):
        self.select_threshold = select_threshold
        self.split_threshold = split_threshold
        self.split_factor = split_factor
        self.ratio = ratio
        self.posting_vector_limit = posting_vector_limit
        self.replica_count = replica_count
    
    def select_heads_dynamically(
        self,
        tree: BKTree,
        num_vectors: int
    ) -> np.ndarray:
        """
        Select cluster heads dynamically using SPTAG algorithm.
        
        Args:
            tree: BKTree built on data
            num_vectors: Total number of vectors
            
        Returns:
            Array of selected centroid indices
        """
        # If ratio would select all vectors, return all
        target_heads = int(np.round(self.ratio * num_vectors))
        if target_heads >= num_vectors:
            return np.arange(num_vectors)
        
        # Binary search for optimal thresholds
        best_selected = []
        min_diff = 100.0
        best_select_thresh = self.select_threshold
        best_split_thresh = self.split_threshold
        
        for select_thresh in range(2, self.select_threshold + 1):
            l = self.split_factor
            r = self.split_threshold
            
            while l < r - 1:
                split_thresh = (l + r) // 2
                
                # Select heads with current thresholds
                selected = []
                self._select_internal(
                    tree, 0, select_thresh, split_thresh, selected
                )
                
                # Remove duplicates
                selected = list(set(selected))
                
                # Calculate difference from target ratio
                diff = len(selected) / num_vectors - self.ratio
                
                if abs(diff) < min_diff:
                    min_diff = abs(diff)
                    best_selected = selected
                    best_select_thresh = select_thresh
                    best_split_thresh = split_thresh
                
                if diff > 0:
                    l = (l + r) // 2
                else:
                    r = (l + r) // 2
        
        print(f"  Dynamic selection: select_thresh={best_select_thresh}, "
              f"split_thresh={best_split_thresh}")
        print(f"  Selected {len(best_selected)} heads "
              f"({len(best_selected)/num_vectors*100:.2f}% of data)")
        
        return np.array(sorted(best_selected))
    
    def _select_internal(
        self,
        tree: BKTree,
        node_id: int,
        select_thresh: int,
        split_thresh: int,
        selected: List[int]
    ) -> int:
        """
        Recursive internal selection (replicates SelectHeadDynamicallyInternal).
        
        Returns:
            Number of vectors in subtree (0 if selected as head)
        """
        node = tree.nodes[node_id]
        children = []
        children_size = 1
        
        # Process children
        if node.children:
            for child_id in node.children:
                cs = self._select_internal(
                    tree, child_id, select_thresh, split_thresh, selected
                )
                if cs > 0:
                    children.append((child_id, cs))
                    children_size += cs
        
        # If subtree large enough, select as head
        if children_size >= select_thresh:
            # Select this node's center
            selected.append(node.center_id)
            
            # If too large, split by selecting largest children
            if children_size > split_thresh:
                # Sort children by size (largest first)
                children.sort(key=lambda x: x[1], reverse=True)
                
                # Select top children
                select_cnt = int(np.ceil(children_size / split_thresh))
                for i in range(min(select_cnt, len(children))):
                    child_node = tree.nodes[children[i][0]]
                    selected.append(child_node.center_id)
            
            return 0  # Subtree handled
        
        return children_size  # Pass size up
    
    def assign_with_limit(
        self,
        data: np.ndarray,
        centroids: np.ndarray,
        centroid_indices: np.ndarray
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Assign vectors to centroids with posting size limits.
        
        Args:
            data: All vectors (N, D)
            centroids: Selected centroid vectors (K, D)
            centroid_indices: Indices of centroids in data
            
        Returns:
            postings: List of vector indices per centroid
            replica_counts: Number of replicas per vector
        """
        N = len(data)
        K = len(centroids)
        
        # Find top-k nearest centroids for each vector
        print(f"  Assigning {N} vectors to {K} centroids (replica={self.replica_count})...")
        
        # Compute distances (batch for efficiency)
        batch_size = 1000
        assignments = []
        
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch = data[i:end]
            
            # Distances to all centroids
            dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            
            # Top-k nearest centroids
            top_k = np.argsort(dists, axis=1)[:, :self.replica_count]
            assignments.append(top_k)
        
        assignments = np.vstack(assignments)  # (N, replica_count)
        
        # Build posting lists
        postings = [[] for _ in range(K)]
        for vec_id in range(N):
            # Skip if this vector is a centroid
            if vec_id in centroid_indices:
                continue
            
            for centroid_id in assignments[vec_id]:
                postings[centroid_id].append(vec_id)
        
        # Apply posting size limits
        print(f"  Applying posting size limit: {self.posting_vector_limit} vectors/posting")
        
        replica_counts = np.zeros(N, dtype=int)
        truncated = 0
        
        for centroid_id in range(K):
            posting = postings[centroid_id]
            
            if len(posting) > self.posting_vector_limit:
                # Keep only first posting_vector_limit vectors
                kept = posting[:self.posting_vector_limit]
                dropped = posting[self.posting_vector_limit:]
                
                postings[centroid_id] = kept
                truncated += len(dropped)
                
                # Update replica counts
                for vec_id in kept:
                    replica_counts[vec_id] += 1
            else:
                for vec_id in posting:
                    replica_counts[vec_id] += 1
        
        if truncated > 0:
            print(f"  Truncated {truncated} assignments due to posting limits")
        
        # Stats
        posting_sizes = [len(p) for p in postings]
        print(f"  Posting sizes: min={min(posting_sizes)}, "
              f"max={max(posting_sizes)}, avg={np.mean(posting_sizes):.1f}")
        
        replica_dist = np.bincount(replica_counts)
        print(f"  Replica distribution:")
        for i, count in enumerate(replica_dist):
            if count > 0:
                print(f"    {i} replicas: {count} vectors ({count/N*100:.1f}%)")
        
        return postings, replica_counts.tolist()
