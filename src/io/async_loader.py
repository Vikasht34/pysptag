"""Async batch I/O for posting list loading."""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import mmap

class AsyncPostingLoader:
    """Load multiple postings in parallel using threads."""
    
    def __init__(self, mmap_file, posting_offsets, num_threads=4):
        self.mmap = mmap_file
        self.posting_offsets = posting_offsets
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
    
    def load_posting(self, cid):
        """Load a single posting."""
        if cid not in self.posting_offsets:
            return cid, None
        
        offset, length = self.posting_offsets[cid]
        data = self.mmap[offset:offset+length]
        return cid, data
    
    def load_batch(self, centroid_ids):
        """Load multiple postings in parallel."""
        futures = [self.executor.submit(self.load_posting, cid) for cid in centroid_ids]
        results = {}
        for future in futures:
            cid, data = future.result()
            if data is not None:
                results[cid] = data
        return results
    
    def close(self):
        self.executor.shutdown(wait=True)
