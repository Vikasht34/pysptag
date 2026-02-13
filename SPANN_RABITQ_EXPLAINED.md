# SPANN + RaBitQ: End-to-End Workflow

## Overview

SPANN (SParse Approximate Nearest Neighbor) + RaBitQ combines graph-based routing with quantized posting lists for fast, memory-efficient vector search.

## Architecture

```
Query → Entry Points → BKTree+RNG → Top Centroids → Posting Lists (RaBitQ) → Rerank → Results
```

---

## INDEXING (Build Phase)

### Step 1: Clustering
**Goal:** Partition vectors into clusters

```python
# K-means clustering
num_clusters = n_vectors // target_posting_size  # e.g., 1M / 1000 = 1000 clusters
centroids = kmeans(data, k=num_clusters)
```

**Output:** 
- `centroids`: [num_clusters, dim] - cluster centers
- Example: 1M vectors → 1000 clusters of ~1000 vectors each

---

### Step 2: Replication (Multi-assignment)
**Goal:** Assign each vector to multiple nearest centroids for recall

```python
# For each vector, find C nearest centroids (C = replica_count, default 8)
for vector in data:
    nearest_centroids = find_k_nearest(vector, centroids, k=replica_count)
    for centroid_id in nearest_centroids:
        posting_lists[centroid_id].append(vector_id)
```

**Output:**
- `posting_lists[i]`: List of vector IDs assigned to centroid i
- Each vector appears in ~8 posting lists (8× replication)
- Total assignments: n_vectors × replica_count

**Why replication?**
- Single assignment: Query might miss nearest neighbor if it's in wrong cluster
- Multi-assignment: Higher recall, query checks multiple clusters

---

### Step 3: Quantize Posting Lists (RaBitQ)
**Goal:** Compress vectors in each posting list

For each posting list:

```python
for centroid_id in range(num_clusters):
    posting_vectors = data[posting_lists[centroid_id]]  # Get actual vectors
    
    # RaBitQ quantization
    rabitq = RaBitQ(dim=dim, bq=4, metric='L2')  # 4-bit quantization
    
    # Step 3a: Compute centroid (already have it!)
    centroid = centroids[centroid_id]
    
    # Step 3b: Compute residuals
    residuals = posting_vectors - centroid
    
    # Step 3c: Normalize residuals
    norms = ||residuals||  # per vector
    normalized_residuals = residuals / norms
    
    # Step 3d: Quantize normalized residuals
    # For 4-bit: map each dimension to [0, 15]
    codes = quantize(normalized_residuals, bits=4)  # [n_vectors, dim] uint8
    
    # Step 3e: Compute RaBitQ factors (for fast distance estimation)
    # These are precomputed per-vector constants
    F_add[i] = ||residual||^2 + 2*||residual|| * <bar_o, c> / <bar_o, o>
    F_rescale[i] = -2*||residual|| / <bar_o, o>
    
    # Store
    posting_codes[centroid_id] = codes
    posting_rabitqs[centroid_id] = rabitq  # Contains F_add, F_rescale, centroid
```

**Output:**
- `posting_codes[i]`: [n_vectors_in_posting, dim] uint8 - quantized codes
- `posting_rabitqs[i]`: RaBitQ object with F_add, F_rescale, centroid

**Memory savings:**
- Original: 1M × 128 × 4 bytes = 512 MB (float32)
- 4-bit: 1M × 128 × 0.5 bytes = 64 MB (4-bit codes)
- **8× compression!**

---

### Step 4: Build Entry Point Index (BKTree + RNG)
**Goal:** Fast routing to relevant centroids

```python
# Build graph on centroids for fast search
bktree = BKTree(num_trees=1, kmeans_k=32)
bktree.build(centroids)

# Build RNG (Relative Neighborhood Graph) for refinement
rng = RNG(neighborhood_size=32)
rng.build(centroids)
```

**Output:**
- `bktree`: Hierarchical tree for coarse search
- `rng`: Graph connecting nearby centroids

**Why BKTree + RNG?**
- BKTree: Fast approximate search (log time)
- RNG: Refines results by exploring graph neighbors

---

## QUERYING (Search Phase)

### Step 1: Find Entry Points
**Goal:** Identify promising centroids to search

```python
# Use BKTree + RNG to find top centroids
entry_point_ids = bktree_rng_search(query, centroids, k=search_internal_result_num)
# search_internal_result_num = 10 (default)
```

**Output:**
- `entry_point_ids`: [10] - IDs of nearest centroids to query

**Example:**
- Query: [0.1, 0.5, ..., 0.3]
- Nearest centroids: [42, 157, 891, 23, ...]

---

### Step 2: Search Posting Lists (RaBitQ Distance Estimation)
**Goal:** Find candidate vectors in each posting list

For each centroid in entry_point_ids:

```python
for centroid_id in entry_point_ids:
    codes = posting_codes[centroid_id]  # [n_vectors, dim] uint8
    rabitq = posting_rabitqs[centroid_id]
    
    # Step 2a: Compute query residual
    q_residual = query - rabitq.centroid
    
    # Step 2b: Compute G_add (query-dependent constant)
    G_add = ||q_residual||^2  # For L2 metric
    
    # Step 2c: Compute inner product between codes and query residual
    c_B = -(2^4 - 1) / 2 = -7.5  # For 4-bit
    ip = codes · q_residual  # [n_vectors] - dot product per vector
    S_q = sum(q_residual)
    ip_adjusted = ip + c_B * S_q
    
    # Step 2d: RaBitQ distance estimation (NO dequantization!)
    estimated_dist = F_add + G_add + F_rescale * ip_adjusted
    # F_add, F_rescale are precomputed per-vector constants
    
    # Step 2e: Get top-k from this posting
    top_k_indices = argsort(estimated_dist)[:max_check]
    candidates.extend(posting_lists[centroid_id][top_k_indices])
```

**Key insight:** 
- Distance computed using **quantized codes directly** (no dequantization!)
- Fast: Only dot products and scalar operations
- Accurate: RaBitQ formula gives good distance estimates

**Output:**
- `candidates`: [~100-1000] - candidate vector IDs from all postings

---

### Step 3: Reranking
**Goal:** Compute exact distances for final top-k

```python
# Get actual vectors for candidates
candidate_vectors = data[candidates]

# Compute true distances
true_distances = ||candidate_vectors - query||^2  # For L2

# Get final top-k
top_k_indices = argsort(true_distances)[:k]
results = candidates[top_k_indices]
```

**Output:**
- `results`: [k] - final top-k nearest neighbor IDs
- `distances`: [k] - exact distances

---

## Complete Example

### Indexing
```python
# 1M vectors, 128 dimensions
data = load_sift1m()  # [1M, 128]

# Build index
index = SPANNRaBitQReplica(
    dim=128,
    target_posting_size=1000,  # ~1000 vectors per cluster
    replica_count=8,           # 8× replication
    bq=4,                      # 4-bit quantization
    metric='L2'
)
index.build(data)

# Result:
# - 1000 clusters
# - 8M total assignments (1M × 8)
# - 64 MB quantized codes (8× compression)
# - BKTree + RNG on 1000 centroids
```

### Querying
```python
query = np.random.randn(128).astype(np.float32)

# Search
distances, indices = index.search(
    query,
    data,
    k=10,                           # Return top-10
    search_internal_result_num=10,  # Check 10 centroids
    max_check=1000                  # Check 1000 candidates total
)

# Result:
# - Searched 10 centroids (out of 1000)
# - Checked ~1000 candidates (out of 1M)
# - Reranked with exact distances
# - Returned top-10 nearest neighbors
# - Latency: ~10ms
```

---

## Performance Characteristics

### Build Time
- K-means: O(n × k × iterations) ≈ 10-60 seconds for 1M vectors
- Replication: O(n × C × k) ≈ 5 seconds
- Quantization: O(n × dim) ≈ 5 seconds
- BKTree: O(k × log k) ≈ <1 second
- **Total: ~20-70 seconds for 1M vectors**

### Query Time
- Entry point search: O(log k) ≈ <1ms
- Posting search: O(candidates × dim) ≈ 5-10ms
- Reranking: O(candidates × dim) ≈ 2-5ms
- **Total: ~10ms for 1M vectors**

### Memory
- Original: n × dim × 4 bytes
- SPANN+RaBitQ: n × dim × (bq/8) bytes + overhead
- **4-bit: 8× compression**
- **2-bit: 16× compression**
- **1-bit: 32× compression**

### Recall
- 1-bit: ~80% recall @ 10ms
- 2-bit: ~90% recall @ 10ms
- 4-bit: ~95% recall @ 10ms
- No-quant: ~99% recall @ 20ms

---

## Key Advantages

1. **Fast**: Graph-based routing + quantized search
2. **Memory-efficient**: 8-32× compression with quantization
3. **Accurate**: RaBitQ formula gives good distance estimates
4. **Scalable**: Billion-scale datasets with <2ms latency
5. **Flexible**: Supports L2, IP, Cosine metrics

---

## Implementation Files

- `src/index/spann_rabitq_replica.py`: Main SPANN index
- `src/quantization/rabitq.py`: RaBitQ quantization
- `src/core/bktree.py`: BKTree for entry points
- `src/core/rng.py`: RNG for graph refinement
