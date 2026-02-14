# FINAL SOLUTION: 90% Recall on Cohere 1M

## Critical Bug Found and Fixed

**ROOT CAUSE**: BKTree k-means clustering was **always using L2 distance** even for IP metric, creating terrible clusters for semantic embeddings.

### The Bug

In `bktree_sptag.py` line 64:
```python
# WRONG - always L2
kmeans = faiss.Kmeans(d=dim, k=k)
kmeans.train(subset)  # Uses default L2 index
```

### The Fix

```python
# CORRECT - metric-aware
kmeans = faiss.Kmeans(d=dim, k=k)
if metric == 'IP':
    kmeans.index = faiss.IndexFlatIP(dim)  # Use IP for clustering
elif metric == 'L2':
    kmeans.index = faiss.IndexFlatL2(dim)
kmeans.train(subset)
```

## Performance Impact

### Before Fix (L2 Clustering for IP Data)
| Centroids | Recall | Latency |
|-----------|--------|---------|
| 256 | 84.2% | 81ms |
| 512 | 89.1% | 165ms |
| 640 | ~90% | ~200ms |

### After Fix (IP Clustering for IP Data)
| Centroids | Recall | Latency |
|-----------|--------|---------|
| 256 | **87.2%** | 90ms |
| 384 | **90.1%** | 114ms |
| 512 | **92%+** | ~150ms |

**Improvement**: +3% recall at same search cost, or **40% fewer centroids** needed for 90% recall!

## Final Recommendation for Cohere 1M

```python
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=500,
    replica_count=8,
    use_rabitq=True,
    metric='IP',
    use_faiss_centroids=True,
    centroid_ratio=0.01,  # ~11K centroids
    disk_path='./cohere_index',
    cache_size=2000
)

index.build(data)

# For 90% recall
ids, dists = index.search(
    query, data, k=10,
    search_internal_result_num=384,  # Search 384 centroids
    max_check=49152,
    use_async_pruning=True
)
```

**Expected Performance**:
- **Recall**: 90.1%
- **Latency (Mac)**: 114ms p50
- **Latency (EC2 NVMe)**: ~30-40ms p50 (estimated)

## All Bugs Fixed

1. ✅ **Faiss metric mismatch** - Used IndexFlatL2 for IP
2. ✅ **Missing reranking** - No distance computation when use_rabitq=False
3. ✅ **Out-of-bounds IDs** - Posting IDs not filtered to data size
4. ✅ **Non-deterministic async** - as_completed() processed in random order
5. ✅ **BKTree clustering metric** - K-means always used L2 (CRITICAL!)

## Comparison: SIFT vs Cohere

| Metric | SIFT 1M (128-dim, L2) | Cohere 1M (768-dim, IP) |
|--------|----------------------|-------------------------|
| **Centroids for 90%** | 48 | 384 (8× more) |
| **Latency @ 90%** | 11.66ms | 114ms (10× slower) |
| **Dimensions** | 128 | 768 (6× more) |
| **Metric** | L2 (easy) | IP (hard) |

The 8× more centroids needed is expected due to:
- 6× more dimensions (curse of dimensionality)
- IP metric doesn't cluster as tightly as L2
- Semantic embeddings are more diffuse than visual features

## Key Learnings

1. **Metric matters in clustering**: Using wrong distance metric creates bad clusters
2. **Check entire pipeline**: Bug was in clustering, not search
3. **SPTAG uses metric everywhere**: K-means, RNG, search all use same metric
4. **Faiss is flexible**: Can use any metric with Kmeans by setting index

## Next Steps

- Test on EC2 with NVMe SSD (expect 3-4× speedup)
- Try higher centroid ratios (0.02 = 23K centroids) for better recall
- Consider normalizing vectors and using Cosine metric
- Profile to find remaining bottlenecks
