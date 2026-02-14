# BKT+RNG vs Faiss for IP Metric

## Summary

BKT+RNG search works correctly for IP metric, but performance degrades with high centroid counts due to auto-calculated posting size limits.

## Test Results

### Small Index (10K vectors, 242 centroids)
- **BKT+RNG**: 100% recall ✅
- **Posting size**: 500 (manual)
- **Avg posting**: 41.3 vectors

### Large Index with 1% Centroids (1M vectors, 10,976 centroids)
- **Faiss**: 84.2% recall @ 81ms ✅
- **Posting size**: 500 (manual)
- **Avg posting**: 86.0 vectors

### Large Index with 2% Centroids (1M vectors, 23,445 centroids)
- **BKT+RNG**: 4.5% recall ❌
- **Faiss**: 2.5% recall ❌
- **Posting size**: 125 (auto-calculated)
- **Avg posting**: 36.4 vectors

## Root Cause

**Auto-calculated posting size becomes too small with high centroid counts:**

```python
# SPTAG formula
posting_size = (posting_vector_limit * page_size) / (dim * value_size)

# With RaBitQ on 768-dim:
# posting_size = (118 * 8192) / (768 * 0.25) = 125 vectors
```

With 23K centroids and posting_size=125:
- Search 256 centroids × 36 avg vectors = **9,216 candidates**
- Not enough to find true neighbors!

With 11K centroids and posting_size=500:
- Search 256 centroids × 86 avg vectors = **22,016 candidates**
- Much better coverage!

## Solution

**Don't use auto-calculated posting size with high centroid ratios:**

```python
# BAD: Auto posting size with 2% centroids
index = SPANNDiskOptimized(
    centroid_ratio=0.02,  # 23K centroids
    # posting_size auto = 125 (too small!)
)

# GOOD: Manual posting size
index = SPANNDiskOptimized(
    centroid_ratio=0.02,
    target_posting_size=500,  # Force larger postings
)
```

## BKT+RNG IP Metric Implementation

The BKT+RNG search correctly handles IP metric:

```python
def compute_dist(a, b):
    if metric == 'IP':
        return -np.dot(a, b)  # Negate for min-heap

# Priority queue uses negated distances
# Smaller (more negative) = better match
# final_results.sort() gives correct order
```

## Recommendation

For IP metric on Cohere 1M:
1. **Use Faiss** for centroid search (`use_faiss_centroids=True`)
2. **Use 1% centroid ratio** (10-11K centroids)
3. **Use posting_size=500** (not auto-calculated)
4. **Search 256-512 centroids** for 84-89% recall

BKT+RNG works but Faiss is faster for large centroid counts.
