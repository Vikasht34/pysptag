"""
Version Map - Track vector versions for SPFresh updates

Based on Microsoft SPTAG VersionLabel.h
"""


class VersionMap:
    """
    Track vector versions for dynamic updates
    
    Each vector has:
    - 7-bit version number (0-127)
    - 1-bit deleted flag
    
    Stored as single uint8 per vector.
    """
    
    def __init__(self):
        self.labels = {}  # vectorID -> uint8
    
    def is_deleted(self, vector_id: int) -> bool:
        """Check if vector is marked as deleted"""
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
    
    def undelete(self, vector_id: int):
        """Unmark vector as deleted"""
        if vector_id in self.labels:
            self.labels[vector_id] &= 0x7F
    
    def update_version(self, vector_id: int):
        """Increment version number"""
        if vector_id not in self.labels:
            self.labels[vector_id] = 1
        else:
            version = (self.labels[vector_id] & 0x7F) + 1
            deleted = self.labels[vector_id] & 0x80
            self.labels[vector_id] = deleted | (version & 0x7F)
    
    def set_version(self, vector_id: int, version: int):
        """Set specific version number"""
        if vector_id not in self.labels:
            self.labels[vector_id] = version & 0x7F
        else:
            deleted = self.labels[vector_id] & 0x80
            self.labels[vector_id] = deleted | (version & 0x7F)
    
    def count(self) -> int:
        """Total number of tracked vectors"""
        return len(self.labels)
    
    def count_deleted(self) -> int:
        """Number of deleted vectors"""
        return sum(1 for label in self.labels.values() if label & 0x80)
    
    def count_active(self) -> int:
        """Number of active (non-deleted) vectors"""
        return self.count() - self.count_deleted()
