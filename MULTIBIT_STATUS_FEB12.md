# Multi-bit RaBitQ Status - Feb 12, 2026

## Current Implementation

### What Works
- ✅ 1-bit quantization: 80-85% recall on SIFT1M
- ✅ best_rescale_factor optimization implemented
- ✅ Sign+magnitude encoding (1 bit sign + (B-1) bits magnitude)
- ✅ All metrics supported (L2, IP, Cosine)

### What's Broken
- ❌ 2-bit: 37-40% recall (expected 90%)
- ❌ 4-bit: 40-54% recall (expected 95%)

## Root Cause

Multi-bit quantization uses **sign+magnitude encoding** but we're still using the **1-bit distance estimation formula**. The official RaBitQ library has separate logic for multi-bit distance estimation.

## Official RaBitQ Multi-bit Approach

From `/Users/viktari/RaBitQ-Library/include/rabitqlib/quantization/rabitq_impl.hpp`:

### Build Phase (lines 534-574)
```cpp
// 1. Compute binary code (sign)
one_bit_code(data, centroid, dim, binary_code);

// 2. Compute ex_bits code (magnitude with optimization)
ex_bits_code_with_factor(data, centroid, dim, ex_bits, total_code, ...);

// 3. Merge: total_code[i] = magnitude + (sign << ex_bits)
for (i = 0; i < dim; ++i) {
    total_code[i] += binary_code[i] << ex_bits;
}

// 4. Compute cb for merged code
float cb = -((1 << ex_bits) - 0.5F);
```

### Key Differences from Our Implementation

1. **ex_bits_code_with_factor**: We haven't implemented this - it computes F_add, F_rescale differently for multi-bit
2. **Delta rescaling**: Official library uses `delta = norm_data / norm_quan * cos_similarity` (lines 558-574)
3. **Separate factor computation**: Multi-bit has different F_add/F_rescale formulas than 1-bit

## What We Need to Implement

### Option 1: Full RaBitQ+ (Complex)
Implement `ex_bits_code_with_factor` from lines 400-530:
- Computes optimal quantization with `best_rescale_factor`
- Computes F_add, F_rescale, F_error for multi-bit
- Uses different formula than 1-bit

### Option 2: Scalar Quantization Mode (Simpler)
Use `rabitq_scalar_impl` from lines 536-574:
- Computes `delta` rescaling factor
- Simpler distance estimation: `delta * (codes · query + vl)`
- Might be faster to implement and debug

## Recommendation

**Implement Option 2 (Scalar Quantization)** first:
1. Compute `delta` and `vl` during build
2. Use simplified distance formula: `estimated_dist = delta * (ip + vl)`
3. Test if this achieves 90%+ recall
4. If not, fall back to Option 1

## Files to Reference

- `/Users/viktari/RaBitQ-Library/include/rabitqlib/quantization/rabitq_impl.hpp`
  - Lines 536-574: `rabitq_scalar_impl` (simpler approach)
  - Lines 400-530: `ex_bits_code_with_factor` (full approach)
  - Lines 273-332: `best_rescale_factor` (already implemented ✓)

## Current Code Location

- `src/quantization/rabitq.py`: RaBitQ implementation
  - `_best_rescale_factor()`: Optimization for multi-bit ✓
  - `_quantize_multibit()`: Sign+magnitude encoding ✓
  - `build()`: Needs multi-bit factor computation ✗
  - `search()`: Needs multi-bit distance formula ✗

## Next Steps

1. Implement scalar quantization mode with `delta` rescaling
2. Test on SIFT1M - target 90%+ recall for 4-bit
3. If successful, document and move to production
4. If not, implement full `ex_bits_code_with_factor`
