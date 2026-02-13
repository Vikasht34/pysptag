# SPANN Optimization Summary

## Goal
Achieve **p90 <10ms latency with 90%+ recall** on EBS disk for high-dimensional vector search.

## Project Structure

```
pysptag/
├── src/
│   ├── clustering/          # NEW: Modular clustering algorithms
│   │   ├── base.py         # Base clustering interface
│   │   ├── kmeans.py       # K-means with posting limits
│   │   └── hierarchical.py # SPTAG-style BKTree clustering
│   ├── core/
│   │   ├── bktree_complete.py  # Complete BKTree implementation
│   │   ├── kdtree.py
│   │   └── rng.py
│   ├── index/
│   │   └── spann_disk_optimized.py  # Pluggable clustering support
│   └── quantization/
│       └── rabitq_numba.py
└── tests/
    ├── test_final_config.py
    └── test_maxcheck_sweep.py
```

## Key Findings

### 1. Quantization Impact
- **No quantization is FASTER** than 4-bit RaBitQ
- With quantization: 5.43ms p50, posting search 69% of time
- Without quantization: 3.32ms p50, reranking only 3.9% of time
- **Recommendation**: Disable quantization for <10ms target

### 2. max_check Parameter
| max_check | p50 (ms) | p90 (ms) | Recall@10 |
|-----------|----------|----------|-----------|
| 2048      | 1.71     | 2.22     | 47.4%     |
| 4096      | 3.28     | 3.88     | 65.0%     |
| **8192**  | **7.64** | **8.84** | **92.6%** ✅ |
| 16384     | 38.55    | 42.03    | 100.0%    |

**Optimal**: max_check=8192 achieves 92.6% recall with acceptable latency

### 3. Posting Size Limits
SPTAG uses `PostingVectorLimit=118` for billion-scale datasets.

**For 10K vectors**:
- posting_size=118: 77.9% recall ❌ (too aggressive)
- posting_size=200: 83.8% recall ❌ (still low)
- **No limits**: 92.6% recall ✅

**Insight**: Posting limits are for billion-scale, not 10K. Need adaptive limits based on dataset size.

### 4. Clustering Comparison
| Method       | Build Time | Clusters | Recall | Notes                    |
|--------------|------------|----------|--------|--------------------------|
| K-means      | 11s        | 50       | 83.8%  | Fast, simple             |
| Hierarchical | >60s       | ~100     | TBD    | Slow BKTree construction |

**Recommendation**: Use k-means for speed, hierarchical for quality on large datasets

## Current Best Configuration

```python
index = SPANNDiskOptimized(
    dim=768,
    target_posting_size=200,      # Adaptive based on dataset size
    replica_count=8,
    use_rabitq=False,             # No quantization
    metric='L2',
    clustering='kmeans',          # Fast k-means
    cache_size=256
)

# Search parameters
search_internal_result_num=64    # Search top 64 centroids
max_check=8192                   # Rerank 8192 candidates
```

**Results on 10K vectors**:
- p50: 3.53ms ✅
- p90: 3.56ms ✅ (target <10ms)
- Recall: 83.8% ❌ (target 90%)

## Next Steps

### Immediate
1. **Adaptive posting limits**: Scale with dataset size
   ```python
   posting_limit = max(118, int(n / num_clusters * 1.5))
   ```

2. **Test on SIFT1M**: Validate on larger dataset
   ```bash
   python3 test_sift1m_efficient.py
   ```

3. **EC2 + EBS testing**: Measure real disk I/O impact
   ```bash
   python3 test_ec2_cohere.py
   ```

### Optimization Opportunities
1. **SIMD reranking**: Vectorize distance computation
2. **Parallel posting search**: Enable multi-threading (fix Numba conflict)
3. **Better centroid selection**: Hierarchical clustering for quality
4. **Disk layout optimization**: Sequential posting storage

## SPTAG Learnings

### Clustering Algorithm
1. Build BKTree on all data
2. SelectHeadDynamically: Binary search for optimal thresholds
   - select_threshold=6: Min vectors to create cluster
   - split_threshold=25: Split if >25 vectors
   - ratio=0.01: Target 1% of vectors as centroids
3. Assign vectors to top-k nearest centroids (replica_count=8)
4. Apply posting limits, keeping closest vectors

### Search Algorithm
1. Search head index (centroids) → top 64
2. Filter by distance ratio (maxDistRatio=10000)
3. Load posting lists from disk
4. Search postings with quantization (optional)
5. Rerank top candidates with true distances

### Key Parameters
```cpp
PostingVectorLimit = 118        // Max vectors per posting
ReplicaCount = 8                // Replicas per vector
searchInternalResultNum = 64    // Search 64 centroids
maxCheck = 4096                 // Rerank 4096 candidates
EnableDataCompression = false   // No compression by default
```

## References
- SPTAG: https://github.com/microsoft/SPTAG
- Paper: "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search"
- Key files:
  - `SPTAG/AnnService/inc/Core/SPANN/ExtraStaticSearcher.h`
  - `SPTAG/AnnService/src/Core/SPANN/SPANNIndex.cpp`
  - `SPTAG/AnnService/inc/Core/Common/BKTree.h`
