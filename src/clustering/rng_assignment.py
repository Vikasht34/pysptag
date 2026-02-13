"""
RNG-filtered replica assignment (SPTAG's NPA strategy)
"""
import numpy as np
from typing import List, Tuple


def assign_with_rng_filtering_batched(
    data: np.ndarray,
    centroids: np.ndarray,
    replica_count: int = 8,
    candidate_num: int = 64,
    rng_factor: float = 1.0,
    metric: str = 'L2',
    batch_size: int = 10000  # Smaller batches to avoid OOM
) -> Tuple[List[List[int]], np.ndarray]:
    """
    SPTAG-style RNG assignment with memory-efficient faiss search.
    """
    import faiss
    n, dim = data.shape
    k = len(centroids)
    
    postings = [[] for _ in range(k)]
    replica_counts = np.zeros(n, dtype=np.int32)
    
    # Build faiss index for fast candidate search
    if metric == 'L2':
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(centroids.astype(np.float32))
    
    # Process in batches
    num_batches = (n + batch_size - 1) // batch_size
    print(f"    Processing {n} vectors in {num_batches} batches with RNG filtering...")
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch = data[start:end].astype(np.float32)
        
        # Find top candidates using faiss (memory efficient)
        batch_dists, top_k_idx = index.search(batch, candidate_num)
        
        # RNG filtering
        for i in range(len(batch)):
            vec_id = start + i
            selected_centroids = []
            
            for j in range(candidate_num):
                if len(selected_centroids) >= replica_count:
                    break
                
                centroid_id = top_k_idx[i, j]
                query_dist = batch_dists[i, j]
                
                # RNG check: compute distances on-demand
                rng_accepted = True
                for prev_id in selected_centroids:
                    if metric == 'L2':
                        between_dist = np.sum((centroids[centroid_id] - centroids[prev_id]) ** 2)
                    else:
                        between_dist = -np.dot(centroids[centroid_id], centroids[prev_id])
                    
                    if rng_factor * between_dist < query_dist:
                        rng_accepted = False
                        break
                
                if rng_accepted:
                    selected_centroids.append(centroid_id)
                    postings[centroid_id].append(vec_id)
                    replica_counts[vec_id] += 1
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"    Batch {batch_idx + 1}/{num_batches} done ({end}/{n} vectors)")
    
    return postings, replica_counts


def truncate_postings_by_distance(
    postings: List[List[int]],
    data: np.ndarray,
    centroids: np.ndarray,
    max_size: int,
    metric: str = 'L2'
) -> List[List[int]]:
    """Truncate posting lists to max_size by keeping closest vectors."""
    truncated = []
    total_removed = 0
    
    for i, posting in enumerate(postings):
        if len(posting) <= max_size:
            truncated.append(posting)
            continue
        
        # Compute distances and keep closest
        posting_vecs = data[posting]
        centroid = centroids[i]
        
        if metric == 'L2':
            dists = np.sum((posting_vecs - centroid) ** 2, axis=1)
        else:
            dists = -np.dot(posting_vecs, centroid)
        
        # Keep closest max_size vectors
        closest_idx = np.argpartition(dists, max_size)[:max_size]
        truncated.append([posting[idx] for idx in closest_idx])
        total_removed += len(posting) - max_size
    
    if total_removed > 0:
        print(f"  Truncated {total_removed} assignments due to posting limits")
    
    return truncated
