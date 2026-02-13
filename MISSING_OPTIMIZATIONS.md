# Missing Optimizations from SPTAG

## Analysis of SPTAG C++ Implementation

### ✅ Already Implemented
1. **Numba JIT compilation** - 3.4× speedup ✓
2. **Parallel execution** - Multi-core processing ✓
3. **RaBitQ quantization** - 1-bit, 2-bit, 4-bit ✓
4. **Replication** - Multiple posting lists per vector ✓
5. **BKTree + RNG** - Fast centroid search ✓

### 🔄 Missing Optimizations (Priority Order)

#### 1. **Batch Query Processing** (Expected: 1.5-2× speedup)
**Status**: Not implemented
**SPTAG**: Processes multiple queries together for better cache utilization

**Implementation**:
```python
def search_batch(self, queries: np.ndarray, data: np.ndarray, k: int):
    """Search multiple queries at once"""
    batch_size = len(queries)
    all_results = []
    
    # Find centroids for all queries at once (vectorized)
    centroid_dists = np.dot(queries, self.centroids.T)  # (batch, n_centroids)
    
    # Process queries in batch
    for query in queries:
        # ... existing search logic
        pass
    
    return all_results
```

**Benefits**:
- Amortize overhead across queries
- Better CPU cache utilization
- Enable BLAS level-3 operations

---

#### 2. **Early Termination** (Expected: 1.2-1.5× speedup)
**Status**: Partially implemented (max_check parameter)
**SPTAG**: Stops when top-k candidates stabilize

**Current**: Check all posting lists up to max_check
**Optimized**: Stop when top-k hasn't changed for N iterations

**Implementation**:
```python
def search_with_early_termination(self, query, data, k, stability_threshold=3):
    """Stop when top-k stabilizes"""
    top_k_candidates = []
    stable_count = 0
    
    for posting_list in sorted_posting_lists:
        prev_top_k = set([c[1] for c in top_k_candidates[:k]])
        
        # Search posting list
        candidates = self._search_posting_list(posting_list, query)
        top_k_candidates = merge_and_sort(top_k_candidates, candidates)[:k*2]
        
        # Check if top-k changed
        curr_top_k = set([c[1] for c in top_k_candidates[:k]])
        if curr_top_k == prev_top_k:
            stable_count += 1
            if stable_count >= stability_threshold:
                break  # Early termination!
        else:
            stable_count = 0
    
    return top_k_candidates[:k]
```

---

#### 3. **Posting Page Limit** (Expected: 1.2-1.3× speedup)
**Status**: Not implemented
**SPTAG**: Limits vectors read per posting list

**Current**: Read entire posting list
**Optimized**: Read only first N vectors per posting

**Implementation**:
```python
def search(self, query, data, k, max_vectors_per_posting=1000):
    """Limit vectors checked per posting list"""
    for centroid_id in nearest_centroids:
        posting_ids = self.posting_lists[centroid_id]
        
        # Limit vectors per posting
        posting_ids = posting_ids[:max_vectors_per_posting]
        
        # Search limited posting
        candidates = self._search_posting_list(posting_ids, query)
```

**Benefits**:
- Reduce computation for large posting lists
- Better latency predictability
- Trade-off: Slight recall drop

---

#### 4. **Workspace Pooling** (Expected: 1.1-1.2× speedup)
**Status**: Not implemented
**SPTAG**: Reuses pre-allocated workspace objects

**Current**: Allocate arrays on every search
**Optimized**: Pool and reuse workspace memory

**Implementation**:
```python
class WorkspacePool:
    def __init__(self, size=10):
        self.pool = [Workspace() for _ in range(size)]
        self.available = list(range(size))
    
    def acquire(self):
        if self.available:
            idx = self.available.pop()
            return self.pool[idx]
        return Workspace()  # Fallback
    
    def release(self, workspace):
        workspace.clear()
        # Return to pool

# Usage
workspace = pool.acquire()
try:
    results = index.search(query, workspace=workspace)
finally:
    pool.release(workspace)
```

**Benefits**:
- Reduce memory allocation overhead
- Better cache locality
- Lower GC pressure

---

#### 5. **Deduplication with Hash Table** (Expected: 1.1-1.2× speedup)
**Status**: Using Python set (slower)
**SPTAG**: Custom hash table with linear probing

**Current**: `seen = set()` for deduplication
**Optimized**: Pre-allocated hash table

**Implementation**:
```python
@njit
def deduplicate_with_hash(candidates, hash_size=8192):
    """Fast deduplication with fixed-size hash table"""
    hash_table = np.full(hash_size, -1, dtype=np.int32)
    unique_candidates = []
    
    for candidate in candidates:
        idx = candidate % hash_size
        # Linear probing
        while hash_table[idx] != -1 and hash_table[idx] != candidate:
            idx = (idx + 1) % hash_size
        
        if hash_table[idx] == -1:
            hash_table[idx] = candidate
            unique_candidates.append(candidate)
    
    return unique_candidates
```

---

#### 6. **SIMD Distance Computation** (Expected: 2-4× speedup)
**Status**: Numba auto-vectorizes (partial)
**SPTAG**: Explicit AVX2/AVX512 intrinsics

**Current**: Numba JIT with auto-vectorization
**Optimized**: Explicit SIMD with C++ extension

**Implementation** (C++ extension):
```cpp
#include <immintrin.h>

float compute_l2_avx512(const float* a, const float* b, int dim) {
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }
    return _mm512_reduce_add_ps(sum);
}
```

**Benefits**:
- 16 float32 operations per instruction (AVX512)
- 2-4× faster than scalar code
- Requires C++ extension

---

#### 7. **Prefetching** (Expected: 1.1-1.2× speedup)
**Status**: Not implemented
**SPTAG**: Prefetch next posting list while processing current

**Implementation**:
```python
# Requires C++ extension
for i, centroid_id in enumerate(nearest_centroids):
    # Prefetch next posting list
    if i + 1 < len(nearest_centroids):
        next_posting = self.posting_lists[nearest_centroids[i+1]]
        __builtin_prefetch(next_posting)  # C++ intrinsic
    
    # Process current posting
    results = self._search_posting_list(centroid_id, query)
```

---

#### 8. **Compressed Posting Lists** (Memory optimization)
**Status**: Basic quantization only
**SPTAG**: ZSTD compression for disk storage

**Current**: Store quantized codes in memory
**Optimized**: Compress with ZSTD for disk

**Implementation**:
```python
import zstandard as zstd

def compress_posting_list(codes):
    """Compress posting list with ZSTD"""
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.compress(codes.tobytes())

def decompress_posting_list(compressed):
    """Decompress posting list"""
    dctx = zstd.ZstdDecompressor()
    return np.frombuffer(dctx.decompress(compressed), dtype=np.uint8)
```

**Benefits**:
- 2-4× additional compression
- Lower memory usage
- Slower search (decompression overhead)

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Numba JIT** - DONE (3.4× speedup)
2. **Batch query processing** - Easy, 1.5-2× speedup
3. **Early termination** - Medium, 1.2-1.5× speedup
4. **Posting page limit** - Easy, 1.2-1.3× speedup

**Expected combined**: 2-3× additional speedup → **1ms latency**

### Phase 2: Advanced (3-5 days)
5. **Workspace pooling** - Medium, 1.1-1.2× speedup
6. **Hash table deduplication** - Medium, 1.1-1.2× speedup
7. **SIMD with C++ extension** - Hard, 2-4× speedup

**Expected combined**: 2-3× additional speedup → **0.5ms latency**

### Phase 3: Production (1 week)
8. **Prefetching** - Hard, requires C++
9. **Compressed posting lists** - Medium, memory optimization
10. **Disk I/O** - Hard, for billion-scale

---

## Current Status vs Target

| Metric | Current | Phase 1 Target | Phase 2 Target |
|--------|---------|----------------|----------------|
| Latency | 2.5ms | 1ms | 0.5ms |
| Recall | 93% | 93% | 93% |
| QPS | 200 | 500 | 1000 |

---

## Recommendation

**Start with Phase 1** (batch processing, early termination, posting page limit):
- Easy to implement in Python
- 2-3× additional speedup expected
- No C++ required
- Should reach **1ms latency target**

**Then evaluate if Phase 2 is needed** based on production requirements.
