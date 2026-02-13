# SPTAG Algorithm Analysis

## Problem We're Solving
**Goal**: p90 <10ms latency with 90%+ recall on disk-based search

**Current Issue**: 
- With posting limits: 11-18% recall ❌
- Without posting limits: 390MB disk, still low recall
- We're not understanding SPTAG's approach correctly

## SPTAG's Two-Phase Approach

### Phase 1: Index Building (Offline)

```cpp
// From SPANNIndex.cpp BuildIndex()

1. SelectHead (if enabled):
   - Build BKTree on ALL data
   - SelectHeadDynamically() → Select ~1% of vectors as "head" (centroids)
   - These become the in-memory index

2. BuildHead:
   - Build in-memory index (BKT/KDT) on selected heads
   - This is for fast centroid search

3. BuildSSDIndex:
   - For each non-head vector:
     a. Search head index → find top-K nearest heads (K = replicaCount = 8)
     b. Assign vector to those K posting lists
   - Apply PostingVectorLimit (118) by keeping CLOSEST vectors
   - Save posting lists to disk
```

### Phase 2: Search (Online)

```cpp
// From SPANNIndex.cpp SearchIndex()

1. Search head index:
   - Query → top 64 centroids (searchInternalResultNum)
   - Filter by distance ratio (maxDistRatio)

2. Load posting lists from disk:
   - For each of the 64 centroids
   - Load posting list (up to searchPostingPageLimit pages)

3. Search posting lists:
   - Compute distances to vectors in postings
   - Keep top maxCheck (4096) candidates

4. Rerank:
   - Compute TRUE distances to top candidates
   - Return top-K
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| **Building** |
| m_ratio | 0.01 | 1% of vectors as heads |
| m_replicaCount | 8 | Each vector in 8 postings |
| m_postingVectorLimit | 118 | Max vectors per posting |
| **Search** |
| m_searchInternalResultNum | 64 | Search top 64 centroids |
| m_maxCheck | 4096 | Rerank top 4096 candidates |
| m_maxDistRatio | 10000 | Distance filter for centroids |

## Critical Insight: Why PostingVectorLimit = 118?

For **1 billion vectors**:
- 1% heads = 10M centroids
- Each vector in 8 postings
- Total assignments = 1B × 8 = 8B
- Per posting = 8B / 10M = **800 vectors**
- Limit to 118 = keep only closest 15% (118/800)

For **100K vectors** (our test):
- 1% heads = 1K centroids  
- Each vector in 8 postings
- Total assignments = 100K × 8 = 800K
- Per posting = 800K / 1K = **800 vectors**
- Limit to 118 = keep only closest 15%

**But we're using 100 centroids, not 1000!**
- Per posting = 800K / 100 = **8000 vectors**
- Limit to 118 = keep only 1.5% ❌ **This kills recall!**

## What We're Doing Wrong

### Issue 1: Wrong Number of Clusters
```python
# Current (WRONG):
target_posting_size = 1000
num_clusters = 100000 / 1000 = 100  # Too few!

# SPTAG (CORRECT):
ratio = 0.01
num_clusters = 100000 * 0.01 = 1000  # 1% of data
```

### Issue 2: Posting Limits Applied Incorrectly
```python
# Current: Apply limit regardless of cluster count
posting_limit = 118  # Fixed

# SPTAG: Limit is relative to expected posting size
expected_size = (n * replica) / num_clusters
if expected_size > 118:
    # Keep closest 118 vectors
else:
    # No limit needed
```

### Issue 3: Not Using Distance-Based Filtering
SPTAG filters centroids by distance ratio:
```cpp
float limitDist = queryResults[0].Dist * m_maxDistRatio;
for (i = 0; i < 64; ++i) {
    if (res->Dist > limitDist) break;  // Stop if too far
}
```

We search all 64 centroids regardless of distance.

## Action Plan

### Step 1: Fix Cluster Count
Use SPTAG's ratio-based approach:
```python
num_clusters = max(1, int(n * 0.01))  # 1% of data
```

### Step 2: Fix Posting Limits
Only apply if posting size exceeds limit:
```python
expected_posting_size = (n * replica_count) / num_clusters
if expected_posting_size > posting_limit:
    # Truncate to posting_limit, keeping closest
else:
    # No truncation needed
```

### Step 3: Add Distance-Based Centroid Filtering
```python
# After finding top 64 centroids
limit_dist = centroid_dists[0] * max_dist_ratio
valid_centroids = [c for c in top_64 if dists[c] <= limit_dist]
```

### Step 4: Test Incrementally
1. Test with correct cluster count (1000 for 100K)
2. Test with proper posting limits
3. Test with distance filtering
4. Measure recall at each step

## Expected Results

For 100K vectors with SPTAG parameters:
- Clusters: 1000 (1% of data)
- Posting size: ~800 vectors/posting
- With limit 118: Keep closest 15%
- Search 64 centroids × 118 vectors = 7,552 vectors
- Rerank top 4096
- **Expected recall: 90%+** (based on SPTAG paper)

## Next Steps

1. Implement ratio-based clustering
2. Fix posting limit logic
3. Test on SIFT 100K
4. Scale to SIFT 1M
5. Deploy to EC2 + EBS
