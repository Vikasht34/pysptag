"""
Hierarchical clustering using BKTree (SPTAG-style)
"""
import numpy as np
from typing import Tuple, List
from .base import ClusteringAlgorithm
from ..core.bktree_complete import BKTree, BKTNode


class HierarchicalClustering(ClusteringAlgorithm):
    """
    SPTAG-style hierarchical clustering using BKTree.
    Implements SelectHeadDynamically algorithm.
    """
    
    def __init__(
        self,
        select_threshold: int = 6,
        split_threshold: int = 25,
        split_factor: int = 2,
        ratio: float = 0.01,
        kmeans_k: int = 32,
        leaf_size: int = 8,
        metric: str = 'L2'
    ):
        """
        Args:
            select_threshold: Min vectors to create cluster
            split_threshold: Split if >threshold vectors
            split_factor: Split factor for large clusters
            ratio: Target ratio of centroids to vectors (1%)
            kmeans_k: K for k-means in BKTree
            leaf_size: Leaf size for BKTree
            metric: Distance metric
        """
        self.select_threshold = select_threshold
        self.split_threshold = split_threshold
        self.split_factor = split_factor
        self.ratio = ratio
        self.kmeans_k = kmeans_k
        self.leaf_size = leaf_size
        self.metric = metric
    
    def cluster(
        self,
        data: np.ndarray,
        target_clusters: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hierarchical clustering via BKTree dynamic selection.
        
        Args:
            data: (N, D) array
            target_clusters: Ignored, uses ratio instead
            
        Returns:
            centroids: Selected cluster centers
            labels: Cluster assignments
        """
        n, dim = data.shape
        
        # Build BKTree
        print(f"  Building BKTree (k={self.kmeans_k}, leaf={self.leaf_size})...")
        tree = BKTree(
            kmeans_k=self.kmeans_k,
            leaf_size=self.leaf_size,
            num_trees=1,
            samples=min(1000, n),
            metric=self.metric
        )
        tree.build(data)
        print(f"  Built tree with {len(tree.nodes)} nodes")
        
        # Select heads dynamically
        print(f"  Selecting heads dynamically (ratio={self.ratio})...")
        selected_indices = self._select_heads_dynamically(tree, n)
        
        centroids = data[selected_indices].copy()
        
        # Assign labels
        if self.metric == 'L2':
            dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        else:
            dists = -np.dot(data, centroids.T)
        
        labels = np.argmin(dists, axis=1)
        
        print(f"  Selected {len(centroids)} centroids ({len(centroids)/n*100:.2f}% of data)")
        
        return centroids, labels
    
    def _select_heads_dynamically(
        self,
        tree: BKTree,
        num_vectors: int
    ) -> np.ndarray:
        """
        SPTAG's SelectHeadDynamically algorithm.
        Binary search for optimal select/split thresholds.
        """
        target_heads = int(np.round(self.ratio * num_vectors))
        if target_heads >= num_vectors:
            return np.arange(num_vectors)
        
        best_selected = []
        min_diff = 100.0
        best_select_thresh = self.select_threshold
        best_split_thresh = self.split_threshold
        
        # Binary search over thresholds
        for select_thresh in range(2, self.select_threshold + 1):
            l = self.split_factor
            r = self.split_threshold
            
            while l < r - 1:
                split_thresh = (l + r) // 2
                
                selected = []
                for root_idx in tree.tree_roots:
                    self._select_internal(
                        tree, root_idx, select_thresh, split_thresh, selected
                    )
                
                selected = list(set(selected))
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
        
        print(f"    Optimal: select_thresh={best_select_thresh}, "
              f"split_thresh={best_split_thresh}, diff={min_diff*100:.2f}%")
        
        return np.array(sorted(best_selected))
    
    def _select_internal(
        self,
        tree: BKTree,
        node_idx: int,
        select_thresh: int,
        split_thresh: int,
        selected: List[int]
    ) -> int:
        """
        Recursive selection (SPTAG's SelectHeadDynamicallyInternal).
        
        Returns:
            Subtree size (0 if selected as head)
        """
        if node_idx >= len(tree.nodes):
            return 1
        
        node = tree.nodes[node_idx]
        children = []
        children_size = 1
        
        # Process children
        if node.childStart >= 0 and node.childEnd > node.childStart:
            for child_idx in range(node.childStart, node.childEnd):
                cs = self._select_internal(
                    tree, child_idx, select_thresh, split_thresh, selected
                )
                if cs > 0:
                    children.append((child_idx, cs))
                    children_size += cs
        
        # Select if subtree large enough
        if children_size >= select_thresh:
            selected.append(node.centerid)
            
            # Split if too large
            if children_size > split_thresh and children:
                children.sort(key=lambda x: x[1], reverse=True)
                select_cnt = int(np.ceil(children_size / split_thresh))
                
                for i in range(min(select_cnt, len(children))):
                    child_node = tree.nodes[children[i][0]]
                    selected.append(child_node.centerid)
            
            return 0
        
        return children_size
    
    def assign_with_replicas(
        self,
        data: np.ndarray,
        centroids: np.ndarray,
        replica_count: int,
        posting_limit: int
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Assign vectors to multiple centroids with posting limits.
        Follows SPTAG's posting list building.
        """
        n = len(data)
        k = len(centroids)
        
        print(f"  Assigning {n} vectors to {k} centroids (replica={replica_count})...")
        
        # Find top-k nearest centroids
        if self.metric == 'L2':
            dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        else:
            dists = -np.dot(data, centroids.T)
        
        nearest = np.argsort(dists, axis=1)[:, :replica_count]
        
        # Build postings
        postings = [[] for _ in range(k)]
        for vec_id, centroid_ids in enumerate(nearest):
            for cid in centroid_ids:
                postings[cid].append(vec_id)
        
        # Apply posting limits (SPTAG truncates excess)
        replica_counts = np.zeros(n, dtype=int)
        truncated = 0
        
        for cid in range(k):
            if len(postings[cid]) > posting_limit:
                truncated += len(postings[cid]) - posting_limit
                postings[cid] = postings[cid][:posting_limit]
            
            for vec_id in postings[cid]:
                replica_counts[vec_id] += 1
        
        if truncated > 0:
            print(f"  Truncated {truncated} assignments due to posting limits")
        
        # Stats
        sizes = [len(p) for p in postings if len(p) > 0]
        if sizes:
            print(f"  Posting sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        return postings, replica_counts
