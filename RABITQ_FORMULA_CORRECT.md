# RaBitQ Multi-Bit Formula - Correct Implementation

## Key Formula (works for all bit widths)

```
estimated_dist = F_add + G_add + F_rescale * (ip + c_B * S_q)
```

## Critical Variables

### Per-vector (stored):
- `F_add = ||o-c||^2 + 2*||o-c|| * <bar_o, c> / <bar_o, o>`
- `F_rescale = -2 * Delta_x / <bar_o, o>`
- `Delta_x` = rescaling factor (from quantization)
- `<bar_o, o>` = inner product between **normalized quantized** and **normalized residual**

### Per-query (computed):
- `G_add = ||q-c||^2`
- `S_q = sum(q_r')` where `q_r' = P^{-1} * q` (rotated query)
- `c_B = -(2^B - 1) / 2`

### Per-pair (computed):
- `ip = <x_u, q_r'>` = inner product between codes and rotated query

## What We're Missing

1. **Rotation matrix P**: We're not rotating the query!
2. **<bar_o, o> computation**: We need normalized quantized vs normalized residual
3. **Delta_x**: The rescaling factor

## Simplified Version (No Rotation)

If we skip rotation (P = Identity), the formula simplifies:

```python
# Build phase
residuals = data - centroid
norms = ||residuals||  # per vector
normalized = residuals / norms  # normalized residuals

# Quantize
codes = quantize(normalized)  # to [0, 2^B-1]

# Compute factors
bar_o = (codes + c_B) / scale  # dequantized normalized
ip_bar_o_o = dot(bar_o, normalized)  # per vector

F_add = norms^2 + 2*norms * dot(bar_o, centroid) / ip_bar_o_o
F_rescale = -2 * norms / ip_bar_o_o

# Search phase
q_residual = query - centroid
G_add = ||q_residual||^2
S_q = sum(q_residual)

# Per candidate
ip = dot(codes, q_residual)
estimated_dist = F_add + G_add + F_rescale * (ip + c_B * S_q)
```

## Why Our Current Code Fails

Looking at our code, we're computing:
```python
xu_cb = codes + cb  # This is correct
ip_resi_xucb = sum(residuals * xu_cb, axis=1)  # This is WRONG!
```

We should be computing:
```python
# Need normalized residuals!
residual_norms = ||residuals||
normalized_residuals = residuals / residual_norms
bar_o = xu_cb / some_scale  # Need proper normalization
ip_bar_o_o = sum(bar_o * normalized_residuals, axis=1)
```

The key insight: **RaBitQ quantizes NORMALIZED residuals, not raw residuals!**
