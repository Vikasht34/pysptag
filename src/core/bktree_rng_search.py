"""
Combined BKTree + RNG search for fast centroid finding

Strategy (from SPTAG):
1. BKTree provides initial candidates (fast tree traversal)
2. RNG graph refines search (greedy graph traversal)
"""
import numpy as np
from typing import Tuple
import heapq


def bktree_rng_search(
    query: np.ndarray,
    centroids: np.ndarray,
    bktree_nodes: list,
    tree_start: list,
    rng_graph: list,
    k: int,
    metric: str = 'L2',
    initial_candidates: int = 10
) -> np.ndarray:
    """
    Combined BKTree + RNG search
    
    Args:
        query: Query vector
        centroids: Centroid vectors
        bktree_nodes: BKTree nodes
        tree_start: Tree start indices
        rng_graph: RNG graph (list of neighbor arrays)
        k: Number of nearest centroids to find
        metric: Distance metric
        initial_candidates: Number of candidates from BKTree
    
    Returns:
        Indices of k nearest centroids
    """
    def compute_dist(a, b):
        if metric == 'L2':
            return np.sum((a - b) ** 2)
        elif metric in ('IP', 'Cosine'):
            return -np.dot(a, b)
        return 0.0
    
    # Step 1: Get initial candidates from BKTree (fast)
    bktree_candidates = []
    
    for tree_idx in range(len(tree_start)):
        root_idx = tree_start[tree_idx]
        stack = [(root_idx, 0.0)]
        
        while stack:
            node_idx, _ = stack.pop()
            node = bktree_nodes[node_idx]
            
            center_id = node.centerid
            if center_id >= 0 and center_id < len(centroids):
                dist = compute_dist(query, centroids[center_id])
                heapq.heappush(bktree_candidates, (dist, center_id))
            
            if node.childStart >= 0:
                for child_idx in range(node.childStart, node.childEnd):
                    stack.append((child_idx, 0.0))
    
    # Get top initial_candidates from BKTree
    initial_set = set()
    initial_list = []
    while bktree_candidates and len(initial_list) < initial_candidates:
        dist, center_id = heapq.heappop(bktree_candidates)
        if center_id not in initial_set:
            initial_set.add(center_id)
            initial_list.append((dist, center_id))
    
    if not initial_list:
        return np.array([], dtype=np.int32)
    
    # Step 2: Expand using RNG graph (greedy search)
    visited = initial_set.copy()
    candidates = initial_list.copy()
    heapq.heapify(candidates)
    
    checked = 0
    max_check = min(len(centroids), k * 10)  # Check at most 10x k nodes
    
    while candidates and checked < max_check:
        dist, current = heapq.heappop(candidates)
        checked += 1
        
        # Check RNG neighbors
        if current < len(rng_graph):
            neighbors = rng_graph[current]
            for neighbor in neighbors:
                if neighbor < 0 or neighbor >= len(centroids) or neighbor in visited:
                    continue
                
                visited.add(neighbor)
                neighbor_dist = compute_dist(query, centroids[neighbor])
                heapq.heappush(candidates, (neighbor_dist, neighbor))
    
    # Step 3: Get top-k from all visited nodes
    final_results = []
    for center_id in visited:
        dist = compute_dist(query, centroids[center_id])
        final_results.append((dist, center_id))
    
    final_results.sort()
    result_indices = [idx for _, idx in final_results[:k]]
    
    return np.array(result_indices, dtype=np.int32)
