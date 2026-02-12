"""
SPANN with Disk-based Posting Lists
For large-scale datasets with EBS storage
"""
import numpy as np
import hnswlib
import h5py
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


class SPANNDisk:
    """
    SPANN with disk-based posting lists
    
    Memory: Only centroids + SPTAG graph
    Disk: All posting lists (on EBS)
    """
    
    def __init__(
        self,
        dim: int,
        index_dir: str,
        target_posting_size: int = 10000,
        closure_factor: float = 1.5,
    ):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.target_posting_size = target_posting_size
        self.closure_factor = closure_factor
        
        self.centroids = None
        self.sptag_index = None
        self.n_clusters = 0
        
        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.postings_dir = self.index_dir / "postings"
        self.postings_dir.mkdir(exist_ok=True)
        
    def build(self, vectors: np.ndarray):
        """Build index and save to disk"""
        n, d = vectors.shape
        print(f"Building SPANN index for {n:,} vectors")
        print(f"Index directory: {self.index_dir}")
        
        # Step 1: Clustering
        print("\n[Step 1] Hierarchical Balanced Clustering...")
        self.n_clusters = max(1, n // self.target_posting_size)
        self.centroids = self._cluster(vectors, self.n_clusters)
        
        # Step 2: Augment and save posting lists to disk
        print("\n[Step 2] Building and saving posting lists to disk...")
        self._build_and_save_postings(vectors)
        
        # Step 3: Build centroid index
        print("\n[Step 3] Building SPTAG graph on centroids...")
        self._build_centroid_index()
        
        # Save metadata
        self._save_metadata()
        
        print(f"\n✓ Index built and saved to {self.index_dir}")
        
    def _cluster(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """K-means clustering"""
        from scipy.cluster.vq import kmeans2
        
        print(f"  Running k-means for {k} clusters...")
        centroids, labels = kmeans2(vectors, k, minit='points', iter=50)
        centroids = centroids.astype('float32')
        
        counts = np.bincount(labels, minlength=k)
        print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, "
              f"avg={counts.mean():.1f}")
        
        return centroids
    
    def _build_and_save_postings(self, vectors: np.ndarray):
        """Build posting lists with NPA and save to disk"""
        n = len(vectors)
        k = len(self.centroids)
        
        # Track which vectors go to which postings
        posting_assignments = defaultdict(list)
        
        # Process in batches
        batch_size = 10000
        replication_counts = []
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = vectors[start:end]
            
            if start % 50000 == 0:
                print(f"  Processing vectors {start:,} - {end:,}...")
            
            # Compute distances to centroids
            dists = np.sum((batch[:, None, :] - self.centroids[None, :, :])**2, axis=2)
            
            for i in range(len(batch)):
                global_i = start + i
                
                # NPA: Find all centroids within closure_factor × nearest_dist
                nearest_dist = dists[i].min()
                threshold = self.closure_factor * nearest_dist
                close_centroids = np.where(dists[i] <= threshold)[0]
                
                for c in close_centroids:
                    posting_assignments[int(c)].append(global_i)
                
                replication_counts.append(len(close_centroids))
        
        print(f"  Replication: min={min(replication_counts)}, max={max(replication_counts)}, "
              f"avg={np.mean(replication_counts):.2f}")
        
        # Save each posting list to disk
        print(f"  Saving {k} posting lists to disk...")
        posting_sizes = []
        for c_id in range(k):
            posting = posting_assignments.get(c_id, [])
            posting_sizes.append(len(posting))
            
            # Save as numpy array for fast loading
            posting_file = self.postings_dir / f"posting_{c_id:06d}.npy"
            np.save(posting_file, np.array(posting, dtype=np.int32))
        
        print(f"  Posting sizes: min={min(posting_sizes)}, max={max(posting_sizes)}, "
              f"avg={np.mean(posting_sizes):.1f}")
        print(f"  Total disk usage: {sum(posting_sizes) * 4 / 1024 / 1024:.1f} MB")
    
    def _build_centroid_index(self):
        """Build HNSW on centroids"""
        k = len(self.centroids)
        self.sptag_index = hnswlib.Index(space='l2', dim=self.dim)
        self.sptag_index.init_index(max_elements=k, ef_construction=200, M=16)
        self.sptag_index.add_items(self.centroids, np.arange(k))
        self.sptag_index.set_ef(50)
        
        # Save to disk
        self.sptag_index.save_index(str(self.index_dir / "centroids.hnsw"))
        print(f"  SPTAG index saved")
    
    def _save_metadata(self):
        """Save index metadata"""
        metadata = {
            'dim': self.dim,
            'n_clusters': self.n_clusters,
            'target_posting_size': self.target_posting_size,
            'closure_factor': self.closure_factor,
            'centroids': self.centroids,
        }
        with open(self.index_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  Metadata saved")
    
    def load(self):
        """Load index from disk"""
        print(f"Loading index from {self.index_dir}...")
        
        # Load metadata
        with open(self.index_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.dim = metadata['dim']
        self.n_clusters = metadata['n_clusters']
        self.target_posting_size = metadata['target_posting_size']
        self.closure_factor = metadata['closure_factor']
        self.centroids = metadata['centroids']
        
        # Load SPTAG index
        self.sptag_index = hnswlib.Index(space='l2', dim=self.dim)
        self.sptag_index.load_index(str(self.index_dir / "centroids.hnsw"))
        self.sptag_index.set_ef(50)
        
        print(f"✓ Index loaded: {self.n_clusters} clusters")
    
    def search(self, query: np.ndarray, vectors: np.ndarray, k: int = 10, n_probe: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with query-aware dynamic pruning
        
        Paper Section 3.3: Query-aware Dynamic Pruning
        - Find n_probe nearest centroids
        - Prune postings that cannot contain top-k based on distance bounds
        - Load only necessary postings from disk
        
        Args:
            query: Query vector
            vectors: Full dataset (for computing distances)
            k: Number of neighbors
            n_probe: Number of postings to probe
        """
        query = query.astype('float32').reshape(1, -1)
        
        # Step 1: Find nearest centroids with distances
        centroid_ids, centroid_dists = self.sptag_index.knn_query(query, k=min(n_probe, self.n_clusters))
        centroid_ids = centroid_ids[0]
        centroid_dists = centroid_dists[0]
        
        # Step 2: Query-aware pruning
        # Sort centroids by distance
        sorted_idx = np.argsort(centroid_dists)
        centroid_ids = centroid_ids[sorted_idx]
        centroid_dists = centroid_dists[sorted_idx]
        
        # Track best k distances found so far
        top_k_dists = np.full(k, np.inf, dtype=np.float32)
        candidates = []
        postings_loaded = 0
        
        # Step 3: Load postings with early termination
        for i, (cid, c_dist) in enumerate(zip(centroid_ids, centroid_dists)):
            # Pruning condition: if centroid distance > k-th best distance, skip
            # (all vectors in this posting are at least c_dist away from query)
            if c_dist > top_k_dists[-1]:
                # Can prune remaining postings
                break
            
            # Load posting from disk
            posting_file = self.postings_dir / f"posting_{cid:06d}.npy"
            if posting_file.exists():
                posting = np.load(posting_file)
                candidates.extend(posting.tolist())
                postings_loaded += 1
                
                # Update top-k distances for pruning
                if len(candidates) >= k:
                    # Compute distances for new candidates
                    unique_candidates = list(set(candidates))
                    candidate_vectors = vectors[unique_candidates]
                    distances = np.sum((candidate_vectors - query)**2, axis=1)
                    
                    # Update top-k
                    if len(distances) >= k:
                        top_k_dists = np.partition(distances, k-1)[:k]
                        top_k_dists = np.sort(top_k_dists)
        
        if not candidates:
            return np.array([]), np.array([])
        
        # Step 4: Remove duplicates and compute final distances
        candidates = list(set(candidates))
        candidate_vectors = vectors[candidates]
        distances = np.sum((candidate_vectors - query)**2, axis=1)
        
        # Step 5: Return top-k
        if len(distances) <= k:
            top_k_idx = np.argsort(distances)
        else:
            top_k_idx = np.argpartition(distances, k)[:k]
            top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
        
        result_ids = np.array([candidates[i] for i in top_k_idx])
        result_dists = distances[top_k_idx]
        
        return result_dists, result_ids
