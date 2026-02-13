"""
Memory-efficient RNG assignment using tree search (SPTAG-style)
"""
import numpy as np
from typing import List, Tuple


def assign_with_tree_search(
    data: np.ndarray,
    centroids: np.ndarray,
    tree,  # BKTree or KDTree
    replica_count: int = 8,
    candidate_num: int = 64,
    rng_factor: float = 1.0,
    metric: str = 'L2'
) -> Tuple[List[List[int]], np.ndarray]:
    """
    SPTAG-style assignment: Use tree search to find candidates, then RNG filter.
    Memory-efficient - no NxK distance matrix!
    
    Args:
        data: (N, D) vectors
        centroids: (K, D) cluster centers  
        tree: Tree index built on centroids
        replica_count: Target replicas per vector
        candidate_num: Search top-N candidates via tree
        rng_factor: RNG threshold
        metric: Distance metric
        
    Returns:
        postings: List of posting lists per centroid
        replica_counts: Number of replicas per vector
    """
    n = len(data)
    k = len(centroids)
    
    print(f"    Assigning {n} vectors using tree search (memory-efficient)...")
    
    # Precompute centroid-to-centroid distances (only K x K, not N x K!)
    if metric == 'L2':
        centroid_dists = np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    elif metric in ('IP', 'Cosine'):
        centroid_dists = -np.dot(centroids, centroids.T)
    else:
        centroid_dists = np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    
    postings = [[] for _ in range(k)]
    replica_counts = np.zeros(n, dtype=np.int32)
    
    # Process vectors one-by-one (SPTAG style)
    for vec_id in range(n):
        query = data[vec_id]
        
        # Search tree for top candidates
        candidate_ids, candidate_dists = tree.search(query, k=candidate_num)
        
        # RNG filtering
        selected_centroids = []
        for i in range(len(candidate_ids)):
            if len(selected_centroids) >= replica_count:
                break
            
            centroid_id = candidate_ids[i]
            query_dist = candidate_dists[i]
            
            # RNG check
            rng_accepted = True
            for existing_id in selected_centroids:
                between_dist = centroid_dists[centroid_id, existing_id]
                if rng_factor * between_dist < query_dist:
                    rng_accepted = False
                    break
            
            if rng_accepted:
                selected_centroids.append(centroid_id)
                postings[centroid_id].append(vec_id)
                replica_counts[vec_id] += 1
        
        # Progress
        if (vec_id + 1) % 50000 == 0:
            print(f"    Assigned {vec_id + 1}/{n} vectors...")
    
    return postings, replica_counts
