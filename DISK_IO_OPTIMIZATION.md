# SPANN Disk I/O Optimization Strategy

## Goal
**Minimize disk reads per query** to achieve <2ms latency

## SPANN Paper Key Insights

### 1. **Posting List Pages**
- Posting lists are divided into **fixed-size pages** (e.g., 200 vectors per page)
- Only read the **first few pages** of each posting list
- Don't read the entire posting list!

### 2. **Posting List Rearrangement**
- Vectors in each posting are **sorted by distance to centroid**
- Closest vectors are in the first pages
- This ensures high-quality candidates with minimal disk reads

### 3. **Query-Aware Pruning**
- Dynamically decide which posting lists to visit
- Stop early if we have enough high-quality candidates
- Use distance to centroid as a proxy for quality

## Current Implementation Status

### ✅ Already Implemented:
1. **Posting rearrangement** - Vectors sorted by distance to centroid
2. **Replication** - Each vector in multiple postings (6-8×)
3. **Quantization** - RaBitQ reduces data size by 75-95%

### ❌ NOT Implemented (Critical for Disk):
1. **Page-based reading** - Currently loads entire posting list
2. **Posting page limit** - No limit on vectors read per posting
3. **Iterative search with early stop** - Reads all postings upfront

## What We Need to Add

### 1. Page-Based Posting Storage
```python
# Instead of:
posting_codes[i] = all_codes  # Entire posting

# Do:
posting_pages[i] = [
    codes[0:200],    # Page 0
    codes[200:400],  # Page 1
    codes[400:600],  # Page 2
    ...
]
```

### 2. Limited Page Reads Per Posting
```python
# Current: Read all vectors from posting
codes = self.posting_codes[centroid_id]  # All vectors!

# Should be: Read only first N pages
max_pages = 3  # Read only 600 vectors max
codes = []
for page_id in range(min(max_pages, len(self.posting_pages[centroid_id]))):
    codes.append(self.posting_pages[centroid_id][page_id])
codes = np.concatenate(codes)
```

### 3. Iterative Search with Early Stop
```python
# Current: Visit all centroids upfront
for centroid_id in nearest_centroids[:128]:
    # Read posting
    
# Should be: Iterative with early stop
candidates = []
for centroid_id in nearest_centroids:
    # Read first page only
    page_0 = self.posting_pages[centroid_id][0]
    candidates.extend(search_in_page(page_0))
    
    # Early stop if we have enough good candidates
    if len(candidates) >= k * 10 and quality_good(candidates):
        break
```

## Expected Impact

### Current (In-Memory):
- Read: All vectors from 128 postings
- Data: 128 postings × 5000 vectors × 1 bit = 80 MB
- Time: ~10ms (all in memory)

### With Disk Optimization:
- Read: 3 pages from 10-20 postings
- Data: 20 postings × 3 pages × 200 vectors × 1 bit = 1.5 MB
- Time: <2ms (minimal disk I/O)
- **50× less data read!**

## Implementation Priority

1. **High Priority** (for disk-based):
   - Page-based storage format
   - Page limit per posting (3-5 pages)
   - Iterative search with early stop

2. **Medium Priority** (optimization):
   - Async I/O for parallel page reads
   - Page caching (LRU)
   - Compression per page

3. **Low Priority** (nice to have):
   - Adaptive page limits based on query
   - Page prefetching
   - SIMD for distance computation

## Key Parameters

Based on SPANN paper for billion-scale:
- **Page size**: 200 vectors
- **Pages per posting**: 3-5 (600-1000 vectors)
- **Centroids to visit**: 10-30 (not 128!)
- **Total vectors checked**: 3K-15K (not 640K!)

## Why This Matters

**Current approach:**
- Visit 128 centroids
- Read 5000 vectors per centroid
- Total: 640,000 vectors read
- With quantization: 80 MB

**Optimized approach:**
- Visit 20 centroids (early stop)
- Read 600 vectors per centroid (3 pages)
- Total: 12,000 vectors read
- With quantization: 1.5 MB
- **53× less data!**

This is how SPANN achieves <2ms latency on billion-scale datasets.
