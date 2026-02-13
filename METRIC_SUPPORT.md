# Metric Support - Complete Implementation

## Summary
✅ **Full support for L2, Inner Product (IP), and Cosine metrics across all components**

## Components

### 1. RaBitQ Quantization (`src/quantization/rabitq.py`)

**Build Phase:**
- **L2**: 
  - `F_add = ||o-c||^2 + 2*||o-c|| * <bar_o, c> / <bar_o, o>`
  - `F_rescale = -2*Delta_x / <bar_o, o>`
  
- **IP/Cosine**:
  - `F_add = -<o-c, c> + ||o-c|| * <bar_o, c> / <bar_o, o>`
  - `F_rescale = -Delta_x / <bar_o, o>`

**Search Phase:**
- **L2**: `G_add = ||q-c||^2`
- **IP/Cosine**: `G_add = -<q, c>`

**Formula (all metrics):**
```python
estimated_dist = F_add + G_add + F_rescale * (ip + c_B * S_q)
```

### 2. SPANN Index (`src/index/spann_rabitq_replica.py`)

**K-means Clustering:**
- L2: `||x - centroid||^2`
- IP: `-x · centroid`
- Cosine: `-x · centroid`

**Centroid Assignment:**
- L2: Euclidean distance
- IP: Negative inner product
- Cosine: Negative inner product

**Entry Point Selection:**
- L2: Euclidean distance to centroids
- IP: Negative inner product with centroids
- Cosine: Negative inner product with centroids

**Posting Search (no-quant):**
- L2: `||codes - query||^2`
- IP: `-codes · query`
- Cosine: `-codes · query`

**Reranking:**
- L2: `||data - query||^2`
- IP: `-data · query`
- Cosine: `-data · query`

### 3. RNG (`src/core/rng.py`)

**Distance Computation:**
```python
def _compute_distance(self, vec1, vec2):
    if self.metric == 'L2':
        return np.sum((vec1 - vec2) ** 2)
    elif self.metric == 'IP':
        return -np.dot(vec1, vec2)
    elif self.metric == 'Cosine':
        return -np.dot(vec1, vec2)
```

### 4. BKTree (`src/core/bktree.py`)

Uses RNG internally, inherits metric support automatically.

## Validation

All metrics tested with:
- 1K vectors, 128 dimensions
- Quantized (1-bit, 2-bit, 4-bit)
- No-quantization baseline
- 10/10 recall achieved for all metrics

**Test Results:**
```
✓ L2 metric PASSED (10/10)
✓ IP metric PASSED (10/10)
✓ Cosine metric PASSED (10/10)
```

## References

- RaBitQ Library: https://vectordb-ntu.github.io/RaBitQ-Library/
- RaBitQ Paper (1-bit): https://arxiv.org/abs/2405.12497
- RaBitQ Paper (multi-bit): https://arxiv.org/abs/2409.09913

## Usage

```python
from src.index.spann_rabitq_replica import SPANNRaBitQReplica

# L2 metric
index_l2 = SPANNRaBitQReplica(dim=128, metric='L2', bq=4)

# Inner Product metric
index_ip = SPANNRaBitQReplica(dim=128, metric='IP', bq=4)

# Cosine metric
index_cosine = SPANNRaBitQReplica(dim=128, metric='Cosine', bq=4)
```

## Implementation Status

| Component | L2 | IP | Cosine | Notes |
|-----------|----|----|--------|-------|
| RaBitQ Build | ✅ | ✅ | ✅ | Correct formulas from paper |
| RaBitQ Search | ✅ | ✅ | ✅ | Metric-specific G_add |
| K-means | ✅ | ✅ | ✅ | Optimized batch computation |
| SPANN Build | ✅ | ✅ | ✅ | All phases metric-aware |
| SPANN Search | ✅ | ✅ | ✅ | Quantized + no-quant paths |
| RNG | ✅ | ✅ | ✅ | Used in BKTree |
| Reranking | ✅ | ✅ | ✅ | True distance computation |

**Status: Production-ready for all metrics!**
