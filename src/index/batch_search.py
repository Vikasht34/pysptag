"""Batch query processing for SPANN"""
import numpy as np
from typing import List, Tuple

def search_batch(index, queries: np.ndarray, data: np.ndarray, k: int = 10,
                 search_internal_result_num: int = 64, max_check: int = 4096) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Search multiple queries in batch for better performance
    
    Benefits:
    - Amortize overhead across queries
    - Better cache utilization
    - Vectorized centroid distance computation
    """
    batch_size = len(queries)
    
    # Step 1: Find nearest centroids for all queries at once (vectorized!)
    if index.metric == 'L2':
        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2<q,c>
        query_norms = np.sum(queries ** 2, axis=1, keepdims=True)  # (batch, 1)
        centroid_norms = np.sum(index.centroids ** 2, axis=1, keepdims=True).T  # (1, n_centroids)
        dots = np.dot(queries, index.centroids.T)  # (batch, n_centroids)
        all_centroid_dists = query_norms + centroid_norms - 2 * dots
    elif index.metric == 'IP':
        all_centroid_dists = -np.dot(queries, index.centroids.T)  # (batch, n_centroids)
    elif index.metric == 'Cosine':
        all_centroid_dists = -np.dot(queries, index.centroids.T)  # (batch, n_centroids)
    
    # Step 2: Get top centroids for each query
    all_nearest_centroids = np.argsort(all_centroid_dists, axis=1)[:, :search_internal_result_num]
    
    # Step 3: Search each query (still sequential, but with shared centroid computation)
    results = []
    for i, query in enumerate(queries):
        nearest_centroids = all_nearest_centroids[i]
        
        # Use existing search logic
        seen = set()
        all_indices = []
        
        for centroid_id in nearest_centroids:
            posting_ids = index.posting_lists[centroid_id]
            if len(posting_ids) == 0:
                continue
            
            codes = index.posting_codes[centroid_id]
            rabitq = index.posting_rabitqs[centroid_id]
            
            search_k = min(max_check, len(posting_ids))
            
            if index.use_rabitq:
                _, local_indices = rabitq.search(query, codes, k=search_k)
            else:
                if index.metric == 'L2':
                    dists = np.sum((codes - query) ** 2, axis=1)
                elif index.metric == 'IP':
                    dists = -np.dot(codes, query)
                elif index.metric == 'Cosine':
                    dists = -np.dot(codes, query)
                local_indices = np.argsort(dists)[:search_k]
            
            for local_idx in local_indices:
                global_id = posting_ids[local_idx]
                if global_id not in seen:
                    seen.add(global_id)
                    all_indices.append(global_id)
                    if len(all_indices) >= max_check:
                        break
            
            if len(all_indices) >= max_check:
                break
        
        # Rerank
        if len(all_indices) == 0:
            results.append((np.array([]), np.array([])))
            continue
        
        candidates = data[all_indices]
        if index.metric == 'L2':
            final_dists = np.sum((candidates - query) ** 2, axis=1)
        elif index.metric == 'IP':
            final_dists = -np.dot(candidates, query)
        elif index.metric == 'Cosine':
            final_dists = -np.dot(candidates, query)
        
        top_k_local = np.argsort(final_dists)[:k]
        top_k_dists = final_dists[top_k_local]
        top_k_indices = np.array([all_indices[i] for i in top_k_local])
        
        results.append((top_k_dists, top_k_indices))
    
    return results
