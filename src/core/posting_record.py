"""
Posting Size Record - Track posting sizes for split/merge decisions

Based on Microsoft SPTAG PostingSizeRecord.h
"""


class PostingSizeRecord:
    """
    Track posting list sizes for SPFresh split/merge decisions
    
    Thresholds:
    - Split: size > 2 × target_size
    - Merge: size < 0.5 × target_size
    """
    
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
        """Update posting size by delta"""
        self.sizes[posting_id] = self.sizes.get(posting_id, 0) + delta
    
    def set_size(self, posting_id: int, size: int):
        """Set posting size directly"""
        self.sizes[posting_id] = size
    
    def get_size(self, posting_id: int) -> int:
        """Get posting size"""
        return self.sizes.get(posting_id, 0)
    
    def get_all_sizes(self) -> dict:
        """Get all posting sizes"""
        return self.sizes.copy()
    
    def count_postings(self) -> int:
        """Total number of postings"""
        return len(self.sizes)
    
    def total_vectors(self) -> int:
        """Total vectors across all postings"""
        return sum(self.sizes.values())
    
    def average_size(self) -> float:
        """Average posting size"""
        if not self.sizes:
            return 0.0
        return self.total_vectors() / len(self.sizes)
