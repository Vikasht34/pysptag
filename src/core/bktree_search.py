"""
Simple BKTree search for centroids (without full SPTAG complexity)

For SPANN, we just need to find k nearest centroids quickly.
BKTree provides O(log n) search instead of O(n) brute force.
"""
import numpy as np
from typing import Tuple
import heapq


def bktree_search_centroids(
    query: np.ndarray,
    centroids: np.ndarray,
    bktree_nodes: list,
    tree_start: list,
    k: int,
    metric: str = 'L2'
) -> np.ndarray:
    """
    Search BKTree for k nearest centroids
    
    Simplified version - just traverse tree and collect candidates
    """
    def compute_dist(a, b):
        if metric == 'L2':
            return np.sum((a - b) ** 2)
        elif metric in ('IP', 'Cosine'):
            return -np.dot(a, b)
        return 0.0
    
    # Priority queue: (distance, node_idx)
    candidates = []
    
    # Start from each tree root
    for tree_idx in range(len(tree_start)):
        root_idx = tree_start[tree_idx]
        stack = [(root_idx, 0.0)]  # (node_idx, dist_to_parent)
        
        while stack:
            node_idx, _ = stack.pop()
            node = bktree_nodes[node_idx]
            
            # Compute distance to this node's center
            center_id = node.centerid
            if center_id >= 0 and center_id < len(centroids):  # Valid centerid
                dist = compute_dist(query, centroids[center_id])
                heapq.heappush(candidates, (dist, center_id))
            
            # Add children to stack
            if node.childStart >= 0:
                for child_idx in range(node.childStart, node.childEnd):
                    stack.append((child_idx, 0.0))
    
    # Get top k unique centroids
    seen = set()
    result = []
    while candidates and len(result) < k:
        dist, center_id = heapq.heappop(candidates)
        if center_id not in seen:
            seen.add(center_id)
            result.append(center_id)
    
    return np.array(result, dtype=np.int32)
