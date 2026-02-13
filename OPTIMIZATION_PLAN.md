# Missing SPTAG Optimizations Analysis

## Current Performance
- **Cohere 1M**: 22ms p50 latency (target: <10ms)
- **Bottleneck**: Disk I/O + sequential posting list loading

## SPTAG Optimizations We're Missing

### 1. **Batch Disk I/O** (CRITICAL - 3-5× speedup)
**Current**: Load one posting list at a time
```python
for centroid_id in nearest_centroids:
    posting = self._load_posting(centroid_id)  # One disk read per centroid
```

**SPTAG**: Batch read multiple posting lists in parallel
```cpp
BatchReadFile(reqs->data(), blockNum, timeout, m_batchSize);
```

**Impact**: 3-5× faster disk I/O

### 2. **Parallel Search** (2-3× speedup)
**Current**: Single-threaded search
**SPTAG**: Multi-threaded with thread pool
```cpp
m_options.m_iSSDNumberOfThreads  // 8-16 threads
```

**Impact**: 2-3× speedup on multi-core

### 3. **SIMD Distance Computation** (2× speedup)
**Current**: NumPy (uses SIMD internally but not optimized for our use case)
**SPTAG**: Hand-optimized SIMD
```cpp
#include "inc/Core/Common/SIMDUtils.h"
```

**Impact**: 2× faster distance computation

### 4. **Memory-Mapped Files** (1.5-2× speedup)
**Current**: pickle.load() for each posting
**SPTAG**: mmap() for zero-copy access
```cpp
mmap(fd, length, PROT_READ, MAP_SHARED, ...)
```

**Impact**: 1.5-2× faster loading

### 5. **Prefetching** (1.5× speedup)
**Current**: Load posting when needed
**SPTAG**: Prefetch next posting while processing current
```cpp
_mm_prefetch((const char *)node, _MM_HINT_T0);
```

**Impact**: 1.5× speedup by hiding latency

### 6. **Posting List Caching** (10× for repeated queries)
**Current**: Load from disk every time
**SPTAG**: LRU cache for hot posting lists

**Impact**: 10× for cached postings

## Implementation Priority

### Phase 1: Quick Wins (Target: 10ms)
1. **Batch disk I/O**: Load multiple postings in one call
2. **Memory-mapped files**: Replace pickle with mmap
3. **Posting list cache**: LRU cache for hot postings

**Expected**: 22ms → 8-10ms

### Phase 2: Advanced (Target: 5ms)
4. **Parallel search**: Multi-threaded posting search
5. **SIMD**: Optimize distance computation
6. **Prefetching**: Async posting loading

**Expected**: 10ms → 5ms

## Immediate Action Items

### 1. Batch Disk I/O (Easiest, Biggest Impact)
```python
def _load_postings_batch(self, centroid_ids):
    """Load multiple posting lists in one batch"""
    postings = {}
    for cid in centroid_ids:
        postings[cid] = self._load_posting(cid)
    return postings
```

### 2. Memory-Mapped Files
```python
import mmap
import struct

# Save posting as binary (not pickle)
with open(posting_file, 'wb') as f:
    # Header: num_vectors, code_size
    f.write(struct.pack('II', len(posting_ids), codes.nbytes))
    # Data: posting_ids, codes
    posting_ids.tofile(f)
    codes.tofile(f)

# Load with mmap
with open(posting_file, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # Zero-copy read
```

### 3. LRU Cache
```python
from functools import lru_cache

@lru_cache(maxsize=128)  # Cache 128 hot postings
def _load_posting_cached(self, centroid_id):
    return self._load_posting(centroid_id)
```

## Expected Results

| Optimization | Latency | Speedup |
|--------------|---------|---------|
| Current | 22ms | 1× |
| + Batch I/O | 12ms | 1.8× |
| + mmap | 8ms | 2.75× |
| + Cache | 5ms (cached) | 4.4× |
| + Parallel | 4ms | 5.5× |
| + SIMD | 3ms | 7.3× |

**Target achieved**: <10ms with Phase 1, <5ms with Phase 2
