"""SPTAG-style BKTree implementation using faiss k-means."""
import numpy as np
from dataclasses import dataclass
from typing import List
import faiss
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BKTNode:
    centerid: int
    childStart: int = -1
    childEnd: int = -1


class BKTreeSPTAG:
    """BKTree using SPTAG's exact algorithm."""
    
    def __init__(self, kmeans_k: int = 32, leaf_size: int = 8, num_trees: int = 1):
        self.kmeans_k = kmeans_k
        self.leaf_size = leaf_size
        self.num_trees = num_trees
        self.nodes: List[BKTNode] = []
        self.tree_roots: List[int] = []
        
    def build(self, data: np.ndarray):
        """Build BKTree on data using SPTAG's algorithm."""
        n = len(data)
        indices = np.arange(n, dtype=np.int32)
        
        # Build single tree - root node created with size as centerid (SPTAG does this!)
        root_idx = len(self.nodes)
        self.tree_roots.append(root_idx)
        self.nodes.append(BKTNode(centerid=n))  # n is out of bounds - used for filtering
        
        # Use stack-based building like SPTAG (LIFO)
        stack = [(root_idx, 0, n)]
        
        while stack:
            node_idx, first, last = stack.pop()
            count = last - first
            
            self.nodes[node_idx].childStart = len(self.nodes)
            
            # Leaf case: create one node per vector
            if count <= self.leaf_size:
                for i in range(first, last):
                    self.nodes.append(BKTNode(centerid=indices[i]))
            else:
                # Cluster using faiss k-means (fast C++ implementation)
                k = min(self.kmeans_k, count)
                subset = data[indices[first:last]].astype(np.float32)
                
                # Use simple k-means for small clusters (< 1248 to avoid faiss warnings)
                if len(subset) < 1248:
                    # Simple k-means for small clusters
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=k, n_init=1, max_iter=20, random_state=0)
                    labels = km.fit_predict(subset)
                else:
                    # Faiss k-means for large clusters
                    kmeans = faiss.Kmeans(d=subset.shape[1], k=k, niter=20, verbose=False)
                    kmeans.train(subset)
                    _, labels = kmeans.index.search(subset, 1)
                    labels = labels.flatten()
                
                # Group by cluster
                cluster_groups = [[] for _ in range(k)]
                for i, label in enumerate(labels):
                    cluster_groups[label].append(i)
                
                # Remove empty clusters and sort by size
                cluster_groups = [c for c in cluster_groups if len(c) > 0]
                cluster_groups.sort(key=len, reverse=True)
                
                # Reorder indices by cluster
                new_indices = []
                for cluster in cluster_groups:
                    for i in cluster:
                        new_indices.append(indices[first + i])
                indices[first:last] = new_indices
                
                # Create child nodes - use last element of each cluster as center
                pos = first
                for cluster in cluster_groups:
                    cluster_size = len(cluster)
                    center_id = indices[pos + cluster_size - 1]
                    child_idx = len(self.nodes)
                    self.nodes.append(BKTNode(centerid=center_id))
                    
                    # Recurse if cluster has more than 1 element
                    if cluster_size > 1:
                        stack.append((child_idx, pos, pos + cluster_size - 1))
                    
                    pos += cluster_size
            
            self.nodes[node_idx].childEnd = len(self.nodes)
        
        # Root centerid stays as n (out of bounds) - SPTAG does this!
        # This makes the filter "centerid < root_centerid" select all valid IDs
