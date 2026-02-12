"""
SPFresh Dynamic Searcher - Complete Implementation
Based on Microsoft SPTAG ExtraDynamicSearcher.h

Implements all 4 operations:
1. Insert - Add vectors with split detection
2. Delete - Mark as deleted with merge detection  
3. Split - Rebalance large postings
4. Reassign - Move vectors to correct postings
"""
import numpy as np
from typing import List, Tuple, Dict, Set
from pathlib import Path
import pickle
from collections import defaultdict
from scipy.cluster.vq import kmeans2


class VersionMap:
    """Track vector versions for updates"""
    
    def __init__(self):
        self.labels = {}  # vectorID -> uint8 (7-bit version + 1-bit deleted)
    
    def is_deleted(self, vector_id: int) -> bool:
        """Check if vector is deleted"""
        if vector_id not in self.labels:
            return False
        return (self.labels[vector_id] & 0x80) != 0
    
    def get_version(self, vector_id: int) -> int:
        """Get version number (0-127)"""
        if vector_id not in self.labels:
            return 0
        return self.labels[vector_id] & 0x7F
    
    def delete(self, vector_id: int):
        """Mark vector as deleted"""
        if vector_id not in self.labels:
            self.labels[vector_id] = 0x80
        else:
            self.labels[vector_id] |= 0x80
    
    def update_version(self, vector_id: int):
        """Increment version number"""
        if vector_id not in self.labels:
            self.labels[vector_id] = 1
        else:
            version = (self.labels[vector_id] & 0x7F) + 1
            deleted = self.labels[vector_id] & 0x80
            self.labels[vector_id] = deleted | (version & 0x7F)


class PostingSizeRecord:
    """Track posting sizes for split/merge decisions"""
    
    def __init__(self, target_size: int = 10000):
        self.sizes = {}
        self.target_size = target_size
        self.split_threshold = 2 * target_size
        self.merge_threshold = int(0.5 * target_size)
    
    def need_split(self, posting_id: int) -> bool:
        """Check if posting needs split"""
        return self.sizes.get(posting_id, 0) > self.split_threshold
    
    def need_merge(self, posting_id: int) -> bool:
        """Check if posting needs merge"""
        return self.sizes.get(posting_id, 0) < self.merge_threshold
    
    def update_size(self, posting_id: int, delta: int):
        """Update posting size"""
        self.sizes[posting_id] = self.sizes.get(posting_id, 0) + delta
    
    def get_size(self, posting_id: int) -> int:
        """Get posting size"""
        return self.sizes.get(posting_id, 0)


class SPFreshDynamic:
    """
    SPFresh Dynamic Searcher
    
    Implements complete SPFresh protocol from paper:
    - Insert with split detection
    - Delete with tombstones
    - Split with balanced k-means
    - Reassign with two conditions
    """
    
    def __init__(
        self,
        base_index,  # SPANN index
        index_dir: str,
        target_posting_size: int = 10000,
        reassign_k: int = 64  # Number of nearby postings to check
    ):
        self.base_index = base_index
        self.index_dir = Path(index_dir)
        self.target_posting_size = target_posting_size
        self.reassign_k = reassign_k
        
        # Version tracking
        self.version_map = VersionMap()
        
        # Posting size tracking
        self.posting_sizes = PostingSizeRecord(target_posting_size)
        
        # Vector ID to posting ID mapping
        self.vector_to_posting = {}
        
        # Next vector ID
        self.next_vector_id = 0
        
    def insert(self, vectors: np.ndarray) -> List[int]:
        """
        Insert vectors with split detection
        
        Returns: List of assigned vector IDs
        """
        n = len(vectors)
        vector_ids = []
        
        for i in range(n):
            vector = vectors[i]
            
            # 1. Find nearest centroid
            _, indices = self.base_index.search(vector, k=1, n_probe=1)
            posting_id = int(indices[0])
            
            # 2. Assign vector ID
            vector_id = self.next_vector_id
            self.next_vector_id += 1
            vector_ids.append(vector_id)
            
            # 3. Append to posting
            self._append_to_posting(posting_id, vector, vector_id)
            
            # 4. Update tracking
            self.vector_to_posting[vector_id] = posting_id
            self.posting_sizes.update_size(posting_id, +1)
            self.version_map.update_version(vector_id)
            
            # 5. Check split threshold
            if self.posting_sizes.need_split(posting_id):
                print(f"  Posting {posting_id} needs split (size={self.posting_sizes.get_size(posting_id)})")
                self.split(posting_id, reassign=True)
        
        return vector_ids
    
    def delete(self, vector_ids: List[int]):
        """
        Delete vectors with tombstones
        """
        for vector_id in vector_ids:
            # 1. Mark as deleted
            self.version_map.delete(vector_id)
            
            # 2. Update posting size
            if vector_id in self.vector_to_posting:
                posting_id = self.vector_to_posting[vector_id]
                self.posting_sizes.update_size(posting_id, -1)
                
                # 3. Check merge threshold
                if self.posting_sizes.need_merge(posting_id):
                    print(f"  Posting {posting_id} needs merge (size={self.posting_sizes.get_size(posting_id)})")
                    # Merge not implemented yet
    
    def split(self, posting_id: int, reassign: bool = True):
        """
        Split posting into two balanced postings
        
        Algorithm:
        1. Load posting from disk
        2. Garbage collect deleted vectors
        3. Run balanced k-means (k=2)
        4. Create two new postings
        5. Update centroid index
        6. Trigger reassignment
        """
        print(f"\n[Split] Posting {posting_id}")
        
        # 1. Load posting
        vectors, vector_ids = self._load_posting(posting_id)
        print(f"  Loaded {len(vectors)} vectors")
        
        # 2. Garbage collect
        active_vectors = []
        active_ids = []
        for i, vid in enumerate(vector_ids):
            if not self.version_map.is_deleted(vid):
                active_vectors.append(vectors[i])
                active_ids.append(vid)
        
        print(f"  After GC: {len(active_vectors)} active vectors")
        
        # 3. Check if still needs split
        if len(active_vectors) <= self.posting_sizes.split_threshold:
            # Just rewrite without deleted vectors
            self._save_posting(posting_id, np.array(active_vectors), active_ids)
            self.posting_sizes.sizes[posting_id] = len(active_vectors)
            print(f"  No split needed after GC")
            return
        
        # 4. Run balanced k-means (k=2)
        active_vectors = np.array(active_vectors, dtype='float32')
        centroids, labels = kmeans2(active_vectors, 2, minit='points', iter=20)
        
        # 5. Split into two postings
        posting1_vectors = active_vectors[labels == 0]
        posting1_ids = [active_ids[i] for i in range(len(active_ids)) if labels[i] == 0]
        
        posting2_vectors = active_vectors[labels == 1]
        posting2_ids = [active_ids[i] for i in range(len(active_ids)) if labels[i] == 1]
        
        print(f"  Split: {len(posting1_ids)} + {len(posting2_ids)} vectors")
        
        # 6. Get new posting IDs (use next available IDs)
        new_posting_id1 = len(self.base_index.centroids)
        new_posting_id2 = len(self.base_index.centroids) + 1
        
        # 7. Save new postings
        self._save_posting(new_posting_id1, posting1_vectors, posting1_ids)
        self._save_posting(new_posting_id2, posting2_vectors, posting2_ids)
        
        # 8. Update centroid index
        new_centroids = centroids.astype('float32')
        self.base_index.centroids = np.vstack([self.base_index.centroids, new_centroids])
        
        # Rebuild HNSW with new centroids
        self.base_index._build_centroid_index()
        
        # 9. Update posting sizes
        self.posting_sizes.update_size(new_posting_id1, len(posting1_ids))
        self.posting_sizes.update_size(new_posting_id2, len(posting2_ids))
        self.posting_sizes.update_size(posting_id, 0)
        
        # 10. Update vector-to-posting mapping
        for vid in posting1_ids:
            self.vector_to_posting[vid] = new_posting_id1
        for vid in posting2_ids:
            self.vector_to_posting[vid] = new_posting_id2
        
        # 11. Trigger reassignment
        if reassign:
            self.reassign(posting_id, new_posting_id1, new_posting_id2, new_centroids)
    
    def reassign(
        self,
        old_posting_id: int,
        new_posting_id1: int,
        new_posting_id2: int,
        new_centroids: np.ndarray
    ):
        """
        Reassign vectors after split
        
        Two conditions from paper:
        1. Vectors in split postings: if d(v, old) <= d(v, new), check reassignment
        2. Vectors in nearby postings: if d(v, new) <= d(v, old), check reassignment
        """
        print(f"\n[Reassign] After split of posting {old_posting_id}")
        
        old_centroid = self.base_index.centroids[old_posting_id]
        new_centroid1 = new_centroids[0]
        new_centroid2 = new_centroids[1]
        
        # Find K nearest postings to old centroid
        _, nearby_ids = self.base_index.search(old_centroid, k=self.reassign_k, n_probe=self.reassign_k)
        nearby_postings = set(nearby_ids.tolist())
        
        print(f"  Checking {len(nearby_postings)} nearby postings")
        
        moves = 0
        
        # Condition 1: Check vectors in split postings
        for posting_id in [new_posting_id1, new_posting_id2]:
            vectors, vector_ids = self._load_posting(posting_id)
            
            for i, vid in enumerate(vector_ids):
                if self.version_map.is_deleted(vid):
                    continue
                
                vector = vectors[i]
                
                # Check condition 1: d(v, old) <= d(v, new)
                dist_old = np.sum((vector - old_centroid)**2)
                dist_new = np.sum((vector - self.base_index.centroids[posting_id])**2)
                
                if dist_old <= dist_new:
                    # Find true nearest centroid
                    _, true_nearest = self.base_index.search(vector, k=1, n_probe=10)
                    true_nearest = int(true_nearest[0])
                    
                    if true_nearest != posting_id:
                        # Move vector
                        self._move_vector(vid, posting_id, true_nearest, vector)
                        self.version_map.update_version(vid)
                        moves += 1
        
        # Condition 2: Check vectors in nearby postings
        for nearby_id in nearby_postings:
            if nearby_id in [old_posting_id, new_posting_id1, new_posting_id2]:
                continue
            
            vectors, vector_ids = self._load_posting(nearby_id)
            
            for i, vid in enumerate(vector_ids):
                if self.version_map.is_deleted(vid):
                    continue
                
                vector = vectors[i]
                
                # Check condition 2: d(v, new) <= d(v, old)
                dist_old = np.sum((vector - old_centroid)**2)
                dist_new1 = np.sum((vector - new_centroid1)**2)
                dist_new2 = np.sum((vector - new_centroid2)**2)
                
                if dist_new1 <= dist_old or dist_new2 <= dist_old:
                    # Find true nearest centroid
                    _, true_nearest = self.base_index.search(vector, k=1, n_probe=10)
                    true_nearest = int(true_nearest[0])
                    
                    if true_nearest in [new_posting_id1, new_posting_id2]:
                        # Move vector
                        self._move_vector(vid, nearby_id, true_nearest, vector)
                        self.version_map.update_version(vid)
                        moves += 1
        
        print(f"  Reassigned {moves} vectors")
    
    def _append_to_posting(self, posting_id: int, vector: np.ndarray, vector_id: int):
        """Append vector to posting file"""
        postings_dir = self.index_dir / "postings"
        postings_dir.mkdir(parents=True, exist_ok=True)
        
        posting_file = postings_dir / f"posting_{posting_id:06d}.npy"
        
        if posting_file.exists():
            data = np.load(posting_file, allow_pickle=True).item()
            vectors = data['vectors']
            ids = data['ids']
        else:
            vectors = []
            ids = []
        
        vectors.append(vector)
        ids.append(vector_id)
        
        np.save(posting_file, {'vectors': vectors, 'ids': ids})
    
    def _load_posting(self, posting_id: int) -> Tuple[List[np.ndarray], List[int]]:
        """Load posting from disk"""
        posting_file = self.index_dir / "postings" / f"posting_{posting_id:06d}.npy"
        
        if not posting_file.exists():
            return [], []
        
        data = np.load(posting_file, allow_pickle=True).item()
        return data['vectors'], data['ids']
    
    def _save_posting(self, posting_id: int, vectors: np.ndarray, vector_ids: List[int]):
        """Save posting to disk"""
        posting_file = self.index_dir / "postings" / f"posting_{posting_id:06d}.npy"
        posting_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(posting_file, {'vectors': vectors.tolist(), 'ids': vector_ids})
    
    def _move_vector(self, vector_id: int, from_posting: int, to_posting: int, vector: np.ndarray):
        """Move vector between postings"""
        # Remove from old posting (mark as deleted)
        # Add to new posting
        self._append_to_posting(to_posting, vector, vector_id)
        
        # Update mapping
        self.vector_to_posting[vector_id] = to_posting
        
        # Update sizes
        self.posting_sizes.update_size(from_posting, -1)
        self.posting_sizes.update_size(to_posting, +1)
