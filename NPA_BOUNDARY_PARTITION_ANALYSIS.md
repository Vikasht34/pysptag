# SPTAG Nearest Boundary Partition & NPA Analysis

## Executive Summary

Deep analysis of SPTAG's **Neighborhood Posting Augmentation (NPA)** and **nearest boundary partition** strategy. This is the core algorithm that makes SPANN work for billion-scale search.

---

## 1. Core Algorithm: ApproximateRNG

### What It Does

**Purpose**: Assign each vector to multiple nearest centroids (replicas) with RNG filtering

**Location**: `/Users/viktari/SPTAG/AnnService/src/Core/VectorIndex.cpp` lines 1140-1250

### Algorithm Flow

```cpp
void ApproximateRNG(fullVectors, exceptIDS, candidateNum, selections, 
                    replicaCount, numThreads, numTrees, leafSize, RNGFactor) {
    
    // For each vector in parallel
    for (fullID in fullVectors) {
        if (fullID in exceptIDS) continue;  // Skip head vectors
        
        // 1. Search head index for nearest centroids
        SearchIndex(resultSet);  // Returns candidateNum results
        
        // 2. Apply RNG filtering to select replicas
        currReplicaCount = 0;
        for (i = 0; i < candidateNum && currReplicaCount < replicaCount; i++) {
            centroid = queryResults[i].VID;
            
            // RNG Check: Is this centroid "good enough"?
            rngAccepted = true;
            for (j = 0; j < currReplicaCount; j++) {
                existingCentroid = selections[j].node;
                nnDist = Distance(centroid, existingCentroid);
                
                // RNG condition: Keep if close to query, far from existing
                if (RNGFactor * nnDist < queryResults[i].Dist) {
                    rngAccepted = false;
                    break;
                }
            }
            
            if (rngAccepted) {
                selections[currReplicaCount].node = centroid;
                selections[currReplicaCount].distance = queryResults[i].Dist;
                currReplicaCount++;
            }
        }
    }
}
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `candidateNum` | 64 | Search top-64 nearest centroids |
| `replicaCount` | 8 | Assign each vector to 8 centroids |
| `RNGFactor` | 1.0 | RNG threshold (1.0 = strict, higher = more permissive) |

---

## 2. RNG Filtering Logic

### The RNG Condition

```cpp
// For a new centroid to be accepted:
for each existing_centroid in selected_centroids:
    dist_between_centroids = Distance(new_centroid, existing_centroid)
    dist_query_to_new = Distance(query, new_centroid)
    
    // Reject if centroids are too close relative to query distance
    if (RNGFactor * dist_between_centroids < dist_query_to_new):
        reject()
```

### Why This Works

**Goal**: Select diverse centroids that cover different regions

**Effect**:
- ✅ Keeps centroids that are far apart (diverse coverage)
- ❌ Rejects centroids that are close together (redundant)
- ✅ Ensures query is near boundary between selected centroids

**Example**:
```
Query Q at position (0, 0)
Centroid A at (1, 0) - distance 1.0
Centroid B at (1.1, 0) - distance 1.1
Centroid C at (0, 1) - distance 1.0

RNG Check for B:
  dist(A, B) = 0.1
  dist(Q, B) = 1.1
  RNGFactor * 0.1 = 0.1 < 1.1 ✓ Accept

RNG Check for C:
  dist(A, C) = 1.41
  dist(Q, C) = 1.0
  RNGFactor * 1.41 = 1.41 > 1.0 ✗ Reject (too far from A)
```

---

## 3. Posting List Building

### Step-by-Step Process

**Location**: `/Users/viktari/SPTAG/AnnService/inc/Core/SPANN/ExtraStaticSearcher.h` lines 645-850

```cpp
bool BuildIndex() {
    // Step 1: Search for replicas with RNG filtering
    for (batch in batches) {
        fullVectors = LoadBatch(start, end);
        
        // Search head index for each vector
        ApproximateRNG(fullVectors, emptySet, candidateNum, 
                      selections, replicaCount, numThreads);
        
        // Count posting list sizes
        for (vector in batch) {
            for (replica in vector.replicas) {
                postingListSize[replica.centroid]++;
                selections[vecOffset + resNum].tonode = vectorID;
            }
        }
    }
    
    // Step 2: Sort selections by centroid ID
    SortSelections(selections);  // Group by centroid
    
    // Step 3: Apply posting size limits
    postingSizeLimit = postingPageLimit * PageSize / vectorInfoSize;
    
    for (centroid in centroids) {
        if (postingListSize[centroid] > postingSizeLimit) {
            // Keep only closest vectors
            selectIdx = FindFirstSelection(centroid);
            
            // Drop furthest vectors
            for (dropID = postingSizeLimit; dropID < postingListSize[centroid]; dropID++) {
                tonode = selections[selectIdx + dropID].tonode;
                replicaCount[tonode]--;  // Reduce replica count
            }
            
            postingListSize[centroid] = postingSizeLimit;
        }
    }
    
    // Step 4: Build posting lists on disk
    for (centroid in centroids) {
        postingList = GetVectorsForCentroid(centroid, selections);
        SavePostingToDisk(centroid, postingList);
    }
}
```

### Posting Size Limit Calculation

```cpp
// Calculate limit based on page size
postingPageLimit = 3;  // Default: 3 pages per posting
PageSize = 4096;       // 4KB per page
vectorInfoSize = dim * sizeof(float) + sizeof(int);  // Vector + ID

// Ensure at least postingVectorLimit vectors fit
postingPageLimit = max(postingPageLimit, 
    (postingVectorLimit * vectorInfoSize + PageSize - 1) / PageSize);

// Calculate actual vector limit
postingSizeLimit = postingPageLimit * PageSize / vectorInfoSize;

// Example for 128D float vectors:
// vectorInfoSize = 128 * 4 + 4 = 516 bytes
// postingSizeLimit = 3 * 4096 / 516 = 23 vectors per posting
```

---

## 4. What We're Missing

### ✅ What We Have

1. **Basic replica assignment** - Assign to top-K nearest centroids
2. **Posting size limits** - Truncate large postings
3. **Distance-based filtering** - `max_dist_ratio` parameter

### ❌ What We're Missing

#### 1. RNG Filtering During Assignment ⭐⭐⭐

**Impact**: High - Better boundary coverage

**SPTAG**:
```cpp
// Checks RNG condition for each replica
for (j = 0; j < currReplicaCount; j++) {
    nnDist = Distance(centroid, existing_centroids[j]);
    if (RNGFactor * nnDist < queryDist) {
        reject();  // Too close to existing
    }
}
```

**Us**:
```python
# Just take top-K nearest
nearest = np.argsort(dists, axis=1)[:, :replica_count]
```

**Why It Matters**:
- SPTAG ensures diverse centroid selection
- We might select 8 centroids that are all clustered together
- SPTAG spreads them out for better boundary coverage
- **Expected impact**: 5-10% recall improvement

#### 2. Posting Limit Based on Distance ⭐⭐

**Impact**: Medium - Better quality postings

**SPTAG**:
```cpp
// Selections are sorted by distance
// When truncating, keep closest vectors
for (dropID = postingSizeLimit; dropID < postingListSize; dropID++) {
    // Drop furthest vectors
    replicaCount[tonode]--;
}
```

**Us**:
```python
# Just truncate arbitrarily
if len(postings[cid]) > posting_limit:
    postings[cid] = postings[cid][:posting_limit]
```

**Why It Matters**:
- SPTAG keeps closest vectors to centroid
- We might keep random vectors
- **Expected impact**: 2-5% recall improvement

#### 3. Replica Count Tracking ⭐

**Impact**: Low - Better statistics

**SPTAG**:
```cpp
std::vector<std::atomic_int> replicaCount(fullCount);

// Track how many replicas each vector has
for (vector in vectors) {
    replicaCount[vector] = num_replicas_assigned;
}

// Report statistics
for (i = 0; i <= replicaCount; i++) {
    count_with_i_replicas = ...;
    LOG("Replica Count Dist: %d, %d", i, count_with_i_replicas);
}
```

**Us**:
```python
# We track but don't use it
replica_counts = np.zeros(n, dtype=int)
```

**Why It Matters**:
- Helps debug posting list quality
- Can identify vectors with too few replicas
- **Expected impact**: 0% (just for monitoring)

---

## 5. Implementation Priority

### Priority 1: RNG Filtering During Assignment ⭐⭐⭐

**Effort**: Medium (2-3 days)
**Impact**: High (5-10% recall improvement)

```python
def assign_with_rng_filtering(
    data: np.ndarray,
    centroids: np.ndarray,
    replica_count: int,
    candidate_num: int = 64,
    rng_factor: float = 1.0
) -> List[List[int]]:
    """Assign vectors to centroids with RNG filtering"""
    n = len(data)
    k = len(centroids)
    
    # Find top-candidate_num nearest centroids
    dists = compute_distances(data, centroids)
    nearest = np.argsort(dists, axis=1)[:, :candidate_num]
    
    postings = [[] for _ in range(k)]
    
    for vec_id in range(n):
        selected_centroids = []
        
        for i in range(candidate_num):
            if len(selected_centroids) >= replica_count:
                break
            
            centroid_id = nearest[vec_id, i]
            query_dist = dists[vec_id, centroid_id]
            
            # RNG check against existing selections
            rng_accepted = True
            for existing_id in selected_centroids:
                centroid_dist = compute_distance(
                    centroids[centroid_id], 
                    centroids[existing_id]
                )
                
                if rng_factor * centroid_dist < query_dist:
                    rng_accepted = False
                    break
            
            if rng_accepted:
                selected_centroids.append(centroid_id)
                postings[centroid_id].append(vec_id)
    
    return postings
```

### Priority 2: Distance-Based Posting Truncation ⭐⭐

**Effort**: Low (1 day)
**Impact**: Medium (2-5% recall improvement)

```python
def truncate_postings_by_distance(
    postings: List[List[int]],
    data: np.ndarray,
    centroids: np.ndarray,
    posting_limit: int
) -> List[List[int]]:
    """Truncate postings, keeping closest vectors"""
    for cid in range(len(postings)):
        if len(postings[cid]) <= posting_limit:
            continue
        
        # Compute distances to centroid
        posting_vecs = data[postings[cid]]
        dists = np.sum((posting_vecs - centroids[cid]) ** 2, axis=1)
        
        # Keep closest vectors
        closest_indices = np.argsort(dists)[:posting_limit]
        postings[cid] = [postings[cid][i] for i in closest_indices]
    
    return postings
```

### Priority 3: Replica Count Statistics ⭐

**Effort**: Low (1 hour)
**Impact**: Low (monitoring only)

```python
def report_replica_statistics(replica_counts: np.ndarray):
    """Report replica count distribution"""
    max_replicas = replica_counts.max()
    
    print("Replica Count Distribution:")
    for i in range(max_replicas + 1):
        count = (replica_counts == i).sum()
        print(f"  {i} replicas: {count} vectors ({count/len(replica_counts)*100:.1f}%)")
```

---

## 6. Expected Impact

### Current Implementation

```python
# Simple top-K assignment
nearest = np.argsort(dists, axis=1)[:, :replica_count]

for vec_id, centroid_ids in enumerate(nearest):
    for cid in centroid_ids:
        postings[cid].append(vec_id)
```

**Issues**:
- May select clustered centroids (poor boundary coverage)
- Arbitrary truncation (may keep far vectors)
- No diversity guarantee

**Recall**: 83-93% on 10K-100K vectors

### With RNG Filtering

```python
# RNG-filtered assignment
for vec_id in range(n):
    selected = []
    for candidate in top_64_nearest:
        if rng_check(candidate, selected):
            selected.append(candidate)
            if len(selected) >= replica_count:
                break
```

**Benefits**:
- Diverse centroid selection (good boundary coverage)
- Distance-based truncation (keep closest)
- Guaranteed diversity

**Expected Recall**: 88-98% on 10K-100K vectors (+5-10%)

---

## 7. Testing Plan

### Test 1: RNG Filtering Impact

```bash
cd /Users/viktari/pysptag

# Test with and without RNG filtering
python test_rng_filtering.py

# Expected results:
# - Without RNG: 83-93% recall
# - With RNG: 88-98% recall
# - Improvement: 5-10 percentage points
```

### Test 2: Posting Truncation Quality

```bash
# Compare truncation strategies
python test_posting_truncation.py

# Expected results:
# - Random truncation: 83-93% recall
# - Distance-based: 85-95% recall
# - Improvement: 2-5 percentage points
```

### Test 3: Full Pipeline

```bash
# Test complete implementation
python test_sift1m_with_rng.py

# Expected results on SIFT1M:
# - Current: 90-93% recall
# - With RNG: 95-98% recall
# - Latency: Similar (5-8ms p50)
```

---

## 8. Conclusion

### What We Learned

**SPTAG's NPA Strategy**:
1. Search top-64 nearest centroids
2. Apply RNG filtering to select diverse 8 replicas
3. Sort selections by centroid
4. Truncate postings by distance (keep closest)
5. Build posting lists on disk

**Key Insight**: RNG filtering ensures vectors are assigned to centroids at **cluster boundaries**, not just nearest centroids. This is critical for high recall.

### What We're Missing

| Feature | Priority | Impact | Effort |
|---------|----------|--------|--------|
| RNG filtering | ⭐⭐⭐ | 5-10% recall | 2-3 days |
| Distance truncation | ⭐⭐ | 2-5% recall | 1 day |
| Replica statistics | ⭐ | 0% (monitoring) | 1 hour |

### Recommendation

**Implement RNG filtering first** - This is the core innovation of SPANN and will give us the biggest recall improvement.

**Expected Results**:
- Current: 83-93% recall on 10K-100K
- With RNG: 88-98% recall on 10K-100K
- On SIFT1M: 95-98% recall (vs 90-93% current)

**Next Steps**:
1. Implement RNG filtering (Priority 1)
2. Test on SIFT1M
3. Add distance-based truncation (Priority 2)
4. Benchmark vs SPTAG C++
