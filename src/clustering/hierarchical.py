"""
Hierarchical clustering using BKTree (SPTAG-style)
"""
import numpy as np
from typing import Tuple, List
from .base import ClusteringAlgorithm
from ..core.bktree_sptag import BKTreeSPTAG as BKTree, BKTNode


class HierarchicalClustering(ClusteringAlgorithm):
    """
    SPTAG-style hierarchical clustering using BKTree.
    Implements SelectHeadDynamically algorithm.
    """
    
    def __init__(
        self,
        select_threshold: int = 0,  # Auto-compute from ratio
        split_threshold: int = 0,  # Auto-compute from ratio
        split_factor: int = 0,  # Auto-compute from ratio
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
        
        # Build BKTree on FULL dataset (SPTAG does this!)
        print(f"  Building BKTree on {n} vectors (k={self.kmeans_k}, leaf={self.leaf_size})...")
        tree = BKTree(
            kmeans_k=self.kmeans_k,
            leaf_size=self.leaf_size,
            num_trees=1,
            metric=self.metric  # Pass metric for correct clustering
        )
        tree.build(data)  # Build on FULL data
        print(f"  Built tree with {len(tree.nodes)} nodes")
        
        # Select heads dynamically
        print(f"  Selecting heads dynamically (ratio={self.ratio})...")
        selected_indices = self._select_heads_dynamically(tree, n)
        
        centroids = data[selected_indices].copy()
        
        # SPTAG doesn't assign labels during clustering!
        # Labels are computed on-demand during assignment via tree search
        labels = None  # Not needed
        
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
        
        # Auto-compute thresholds if set to 0 (SPTAG does this)
        select_threshold = self.select_threshold
        split_threshold = self.split_threshold
        split_factor = self.split_factor
        
        if select_threshold == 0:
            select_threshold = min(num_vectors - 1, int(1 / self.ratio))
        if split_threshold == 0:
            split_threshold = min(num_vectors - 1, int(select_threshold * 2))
        if split_factor == 0:
            split_factor = min(num_vectors - 1, int(np.round(1 / self.ratio)))
        
        best_selected = []
        min_diff = 100.0
        best_select_thresh = select_threshold
        best_split_thresh = split_threshold
        
        # Binary search over split_threshold (SPTAG does this)
        l = split_factor
        r = split_threshold
        
        while l < r - 1:
            split_thresh = (l + r) // 2
            
            selected = []
            for root_idx in tree.tree_roots:
                root_centerid = tree.nodes[root_idx].centerid
                self._select_internal(
                    tree, root_idx, select_threshold, split_thresh, selected, root_centerid, split_factor
                )
            
            selected = list(set(selected))
            diff = len(selected) / num_vectors - self.ratio
            
            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_selected = selected
                best_split_thresh = split_thresh
            
            if diff > 0:
                l = (l + r) // 2
            else:
                r = (l + r) // 2
        
        print(f"    Optimal: select_thresh={select_threshold}, "
              f"split_thresh={best_split_thresh}, diff={min_diff*100:.2f}%")
        
        return np.array(sorted(best_selected))
    
    def _select_internal(
        self,
        tree: BKTree,
        node_idx: int,
        select_thresh: int,
        split_thresh: int,
        selected: List[int],
        root_centerid: int,
        split_factor: int
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
                    tree, child_idx, select_thresh, split_thresh, selected, root_centerid, split_factor
                )
                if cs > 0:
                    children.append((child_idx, cs))
                    children_size += cs
        
        # Select if subtree large enough
        if children_size >= select_thresh:
            # SPTAG filters: only select if centerid < root centerid
            if node.centerid < root_centerid:
                selected.append(node.centerid)
            
            # Split if too large
            if children_size > split_thresh and children:
                children.sort(key=lambda x: x[1], reverse=True)
                select_cnt = int(np.ceil(children_size / split_factor))
                
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
        posting_limit: int,
        use_rng_filtering: bool = True
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Assign vectors to multiple centroids with RNG filtering.
        SPTAG-style: Process in batches to avoid OOM.
        """
        n = len(data)
        k = len(centroids)
        
        if use_rng_filtering:
            # Use RNG filtering with BATCHING (SPTAG style)
            from .rng_assignment import assign_with_rng_filtering_batched
            
            print(f"  Assigning {n} vectors with RNG filtering (batched)...")
            postings, replica_counts = assign_with_rng_filtering_batched(
                data, centroids, replica_count,
                candidate_num=min(64, k),
                rng_factor=1.0,
                metric=self.metric,
                batch_size=100000  # Process 100K vectors at a time
            )
            
            # Truncate by distance if needed
            from .rng_assignment import truncate_postings_by_distance
            postings = truncate_postings_by_distance(
                postings, data, centroids, posting_limit, self.metric
            )
            
            # Stats
            sizes = [len(p) for p in postings if len(p) > 0]
            if sizes:
                print(f"  Posting sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
            
            return postings, replica_counts
        
        # Fallback to simple assignment
        print(f"  Assigning {n} vectors to {k} centroids (replica={replica_count})...")
        
        # Find top-k nearest centroids in batches to avoid OOM
        batch_size = 10000
        nearest = np.zeros((n, replica_count), dtype=np.int32)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = data[start:end]
            
            # Use faiss for fast distance computation (avoids large intermediate arrays)
            import faiss
            if self.metric == 'L2':
                index = faiss.IndexFlatL2(centroids.shape[1])
            else:
                index = faiss.IndexFlatIP(centroids.shape[1])
            
            index.add(centroids.astype(np.float32))
            _, indices = index.search(batch.astype(np.float32), replica_count)
            nearest[start:end] = indices
            
            if (end) % 50000 == 0:
                print(f"    Processed {end}/{n} vectors...")
        
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
