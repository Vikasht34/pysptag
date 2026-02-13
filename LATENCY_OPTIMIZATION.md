# SPTAG Latency Optimizations

## Current Performance Issue
- **Latency**: 95ms (target: <15ms)
- **Disk I/O**: 3.3MB per query (too high!)
- **Problem**: Reading entire posting lists from disk

## SPTAG's Key Optimizations

### 1. **Posting Page Limits** (Most Important!)
SPTAG doesn't read entire posting lists - it uses page limits:
- `m_searchPostingPageLimit`: Max pages to read per posting
- `m_postingPageLimit`: Page size for storage
- Default page size: 4KB or 8KB
- **Effect**: Read only first N vectors from each posting, not all

**Our issue**: We read entire posting lists (avg 800 vectors × 4 bytes = 3.2KB per posting)
- With max_check=4096, we read 4096 postings = 13MB!
- But we only use 64 centroids, so 64 × 3.2KB = 205KB (still high)

### 2. **Async I/O with io_uring**
SPTAG uses asynchronous I/O:
```cpp
m_ioThreads = 2  // Parallel I/O threads
GENERIC_READ with async file handles
```
- Loads multiple postings in parallel
- Overlaps I/O with computation

### 3. **Memory-Mapped Files with Prefetch**
- Uses mmap() for zero-copy reads
- Prefetches pages before access
- OS page cache helps with hot data

### 4. **Posting List Truncation**
- `m_postingVectorLimit`: Max vectors per posting (default 200-500)
- Truncate large postings during build
- Trade recall for speed

### 5. **Quantization (RaBitQ)**
- 1-4 bit quantization reduces posting size by 8-32×
- Faster to load and compute distances
- Rerank top candidates with full precision

### 6. **Smaller max_check**
- SPTAG default: max_check=32-128 (not 4096!)
- With good clustering, don't need to check many clusters
- Our 11K clusters with RNG should work with max_check=64

## Recommended Fixes

### Immediate (High Impact):
1. **Reduce max_check to 64-128** (from 4096)
   - With 11K clusters and RNG, this should be enough
   - Reduces I/O from 13MB to 200-400KB

2. **Truncate posting lists to 200-500 vectors**
   - Current: 800 vectors per posting
   - SPTAG uses 200-500
   - Reduces per-posting I/O by 2-4×

3. **Enable quantization (1-2 bit)**
   - Reduces posting size by 16-32×
   - 3.2KB → 100-200 bytes per posting
   - Total I/O: 64 postings × 200 bytes = 12KB

### Medium Term:
4. **Implement async I/O**
   - Use Python's asyncio or threading
   - Load multiple postings in parallel
   - Overlap I/O with computation

5. **Add posting page limits**
   - Read only first N vectors from each posting
   - Stop early if enough candidates found

### Long Term:
6. **Optimize centroid search**
   - Use HNSW or optimized BKTree for head index
   - Current brute-force on 11K centroids is slow

## Expected Results After Fixes

With max_check=64 + truncate to 200 + 2-bit quantization:
- **Disk I/O**: 64 × (200 vectors × 0.5 bytes) = 6.4KB (500× reduction!)
- **Latency**: <10ms on SSD
- **Recall**: 90%+ (RNG ensures quality)

## Implementation Priority

1. **Test with max_check=64** (1 line change)
2. **Reduce target_posting_size to 200** (1 line change)
3. **Enable 2-bit quantization** (already supported)
4. Run benchmark and measure improvement
