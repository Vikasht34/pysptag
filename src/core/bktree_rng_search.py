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
    initial_candidates: int = 100,
    max_check: int = 500
) -> np.ndarray:
    """
    Combined BKTree + RNG search (SPTAG-style with early termination)
    
    Args:
        query: Query vector
        centroids: Centroid vectors
        bktree_nodes: BKTree nodes
        tree_start: Tree start indices
        rng_graph: RNG graph (list of neighbor arrays)
        k: Number of nearest centroids to find
        metric: Distance metric
        initial_candidates: Number of candidates from BKTree
        max_check: Maximum nodes to check (early termination)
    
    Returns:
        Indices of k nearest centroids
    """
    def compute_dist(a, b):
        if metric == 'L2':
            return np.sum((a - b) ** 2)
        elif metric in ('IP', 'Cosine'):
            return -np.dot(a, b)
        return 0.0
    
    # Step 1: Initialize BKTree search with priority queue
    spt_queue = []  # Priority queue for tree nodes
    visited = set()
    
    # Add root nodes to queue
    for tree_idx in range(len(tree_start)):
        root_idx = tree_start[tree_idx]
        node = bktree_nodes[root_idx]
        
        if node.centerid >= 0 and node.centerid < len(centroids):
            dist = compute_dist(query, centroids[node.centerid])
            heapq.heappush(spt_queue, (dist, root_idx))
    
    # Step 2: Search BKTree with early termination
    ng_queue = []  # Candidates for result
    checked_leaves = 0
    
    while spt_queue and checked_leaves < initial_candidates:
        dist, node_idx = heapq.heappop(spt_queue)
        node = bktree_nodes[node_idx]
        
        # If leaf node, add to candidates
        if node.childStart < 0:
            if node.centerid >= 0 and node.centerid < len(centroids) and node.centerid not in visited:
                visited.add(node.centerid)
                heapq.heappush(ng_queue, (dist, node.centerid))
                checked_leaves += 1
        else:
            # Add center to candidates
            if node.centerid >= 0 and node.centerid < len(centroids) and node.centerid not in visited:
                visited.add(node.centerid)
                heapq.heappush(ng_queue, (dist, node.centerid))
            
            # Add children to queue
            for child_idx in range(node.childStart, node.childEnd):
                child_node = bktree_nodes[child_idx]
                if child_node.centerid >= 0 and child_node.centerid < len(centroids):
                    child_dist = compute_dist(query, centroids[child_node.centerid])
                    heapq.heappush(spt_queue, (child_dist, child_idx))
    
    # Step 3: Expand using RNG graph
    checked = len(visited)
    
    while ng_queue and checked < max_check:
        dist, current = heapq.heappop(ng_queue)
        checked += 1
        
        # Check RNG neighbors
        if current < len(rng_graph):
            neighbors = rng_graph[current]
            for neighbor in neighbors:
                if neighbor < 0 or neighbor >= len(centroids) or neighbor in visited:
                    continue
                
                visited.add(neighbor)
                neighbor_dist = compute_dist(query, centroids[neighbor])
                heapq.heappush(ng_queue, (neighbor_dist, neighbor))
    
    # Step 4: Return top-k
    result = []
    temp_queue = []
    while ng_queue and len(result) < k:
        dist, node_id = heapq.heappop(ng_queue)
        result.append(node_id)
        temp_queue.append((dist, node_id))
    
    # Put back remaining for potential future use
    for item in temp_queue:
        heapq.heappush(ng_queue, item)
    
    return np.array(result[:k], dtype=np.int32)
    
    # Step 3: Get top-k from all visited nodes
    final_results = []
    for center_id in visited:
        dist = compute_dist(query, centroids[center_id])
        final_results.append((dist, center_id))
    
    final_results.sort()
    result_indices = [idx for _, idx in final_results[:k]]
    
    return np.array(result_indices, dtype=np.int32)
