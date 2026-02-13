# Latency Optimization Strategy

## Current Performance (SIFT1M, 1M vectors, 128D)
- **2-bit**: 8.6ms p50 latency, 93% recall ✓
- **4-bit**: 9.0ms p50 latency, 93% recall ✓
- **Target**: <2ms p50 latency (production requirement)
- **Gap**: Need 4-5× speedup

## Optimization Roadmap (Priority Order)

### 1. **Numba JIT Compilation** (Expected: 2-3× speedup)
**Status**: Not yet implemented (numba not installed)

**Implementation**:
```python
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_distances_l2(query, codes, centroid, scale, res_min):
    n = codes.shape[0]
    dists = np.empty(n, dtype=np.float32)
    for i in prange(n):
        # Dequantize and compute L2
        dist = 0.0
        for j in range(codes.shape[1]):
            val = codes[i,j] * scale + res_min + centroid[j]
            diff = val - query[j]
            dist += diff * diff
        dists[i] = dist
    return dists
```

**Benefits**:
- JIT compiles to native code
- Parallel execution across cores
- SIMD auto-vectorization
- No Python overhead

**Install**: `pip install numba`

---

### 2. **Batch Query Processing** (Expected: 1.5-2× speedup)
**Status**: Not implemented

**Current**: Process queries one-by-one
**Optimized**: Process 10-100 queries together

**Benefits**:
- Better cache utilization
- Amortize overhead across queries
- Enable BLAS level-3 operations (matrix-matrix)

**Implementation**:
```python
def search_batch(self, queries: np.ndarray, data: np.ndarray, k: int):
    """Search multiple queries at once"""
    # queries: (batch_size, dim)
    # Use matrix operations instead of loops
    pass
```

---

### 3. **SIMD Distance Computation** (Expected: 2-4× speedup)
**Status**: Not implemented (requires C++ extension or Numba)

**Options**:
a) **Numba with explicit SIMD** (easier)
b) **C++ extension with AVX2/AVX512** (faster, more complex)

**From SPTAG**:
- AVX512: 8× float32 operations per instruction
- AVX2: 4× float32 operations per instruction
- SSE: 2× float32 operations per instruction

**Implementation** (C++ extension):
```cpp
// Use SPTAG's DistanceUtils
#include "inc/Core/Common/DistanceUtils.h"

float compute_l2_avx512(const float* a, const float* b, int dim) {
    return SPTAG::COMMON::DistanceUtils::ComputeL2Distance_AVX512(a, b, dim);
}
```

---

### 4. **Early Termination** (Expected: 1.2-1.5× speedup)
**Status**: Partially implemented (max_check parameter)

**Current**: Check all posting lists up to max_check
**Optimized**: Stop when top-k stabilizes

**Implementation**:
```python
def search_with_early_termination(self, query, data, k, confidence=0.95):
    """Stop when top-k hasn't changed for N iterations"""
    top_k = []
    stable_count = 0
    for posting_list in sorted_posting_lists:
        prev_top_k = set(top_k)
        # Search posting list
        if set(top_k) == prev_top_k:
            stable_count += 1
            if stable_count > threshold:
                break  # Early termination
        else:
            stable_count = 0
    return top_k
```

---

### 5. **Memory Layout Optimization** (Expected: 1.2-1.5× speedup)
**Status**: Not implemented

**Current**: Python lists and dicts
**Optimized**: Contiguous NumPy arrays

**Benefits**:
- Better cache locality
- Prefetching works better
- Reduced pointer chasing

**Implementation**:
```python
# Store posting lists as single contiguous array
self.posting_data = np.concatenate(all_posting_lists)
self.posting_offsets = np.cumsum([0] + [len(pl) for pl in posting_lists])

# Access: posting_data[offsets[i]:offsets[i+1]]
```

---

### 6. **Multi-threading** (Expected: 2-4× speedup on 4+ cores)
**Status**: Not implemented

**Current**: Single-threaded search
**Optimized**: Parallel search across posting lists

**Implementation**:
```python
from concurrent.futures import ThreadPoolExecutor

def search_parallel(self, query, data, k, num_threads=4):
    """Search posting lists in parallel"""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(self._search_posting_list, query, pl) 
                   for pl in posting_lists]
        results = [f.result() for f in futures]
    return merge_results(results, k)
```

---

## Combined Expected Speedup

**Conservative estimate** (multiplicative):
- Numba JIT: 2×
- Batch processing: 1.5×
- Early termination: 1.2×
- Memory layout: 1.2×
- **Total: 2 × 1.5 × 1.2 × 1.2 = 4.3× speedup**

**Optimistic estimate** (with SIMD + multi-threading):
- SIMD (AVX512): 3×
- Multi-threading (4 cores): 3×
- Batch processing: 1.5×
- **Total: 3 × 3 × 1.5 = 13.5× speedup**

**Target**: 8.6ms → 2ms = 4.3× speedup ✓ (achievable with conservative optimizations)

---

## Implementation Priority

1. **Start with Numba JIT** (biggest bang for buck, easiest)
   - Install: `pip install numba`
   - Add `@njit` decorators to hot functions
   - Test on SIFT1M

2. **Add batch processing** (if still not fast enough)
   - Modify search API to accept multiple queries
   - Use matrix operations

3. **Consider C++ extension** (if need <1ms latency)
   - Use SPTAG's SIMD distance functions
   - Requires build system setup

4. **Add multi-threading** (for high QPS scenarios)
   - Use ThreadPoolExecutor or multiprocessing
   - Good for server deployments

---

## Quick Wins (No Code Changes)

1. **Use smaller max_check**: 200 → 100 (2× faster, slight recall drop)
2. **Use fewer replicas**: 4× → 2× (2× faster, slight recall drop)
3. **Use 1-bit quantization**: 8.6ms → 5.2ms (1.6× faster, 82% recall)

---

## Next Steps

Run this to install Numba and test:
```bash
pip install numba
python3 test_numba_optimization.py
```

Expected result: **2-3× speedup** with minimal code changes!
