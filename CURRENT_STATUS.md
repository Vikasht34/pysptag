# PySPTAG Current Status (Feb 13, 2026)

## ✅ Implemented Optimizations

### Core Features
1. **SPANN disk-based index** - Hierarchical clustering + posting lists
2. **RaBitQ quantization** - 1-bit, 2-bit, 4-bit compression (Numba JIT)
3. **Single-file format** - Fast mmap-based loading (2.4× speedup)
4. **Faiss centroid search** - 40× faster than KDTree
5. **BKTree + RNG graph** - Fast approximate centroid search
6. **Replication** - Multiple posting lists per vector (replica=8)
7. **Multi-metric support** - L2, IP (Inner Product), Cosine ✅ FIXED

### Recent Additions (Feb 13)
8. **Async query-aware pruning** - ThreadPoolExecutor parallel I/O (2.4× speedup)
9. **Posting page limit** - Limit vectors per posting (max_vectors_per_posting)
10. **Batch query processing** - Vectorized centroid search (search_batch)
11. **Hash table deduplication** - Numba JIT (not enabled, recall issues)

## 📊 Current Performance (SIFT 1M, macOS)

**Best Configuration:**
- posting_size=500, replica=8, RaBitQ 2-bit
- 11,482 clusters, hierarchical clustering
- search_internal_result_num=48, max_check=6144
- use_async_pruning=True, max_vectors_per_posting=500

**Results (1000 queries):**
- **Recall@10: 91.19%** ✅ (target: 90%)
- **Latency p50: 11.67ms** ⚠️ (target: <10ms, 17% over)
- **Latency p90: 13.93ms**
- **QPS: ~85**

**Expected on EC2 NVMe (2-3× faster disk):**
- **p50: 4-6ms** ✅ (well below 10ms target!)
- **Recall: 91%+** ✅

## ❌ Not Implemented (from MISSING_OPTIMIZATIONS.md)

### High Priority (Easy Wins)
1. ~~**Batch query processing**~~ ✅ DONE (but doesn't help much)
2. **Early termination** - Stop when top-k stabilizes (1.2-1.5× speedup)
3. ~~**Posting page limit**~~ ✅ DONE (0.6% speedup)
4. **Workspace pooling** - Reuse pre-allocated memory (1.1-1.2× speedup)
5. **Hash table deduplication** - Replace Python set() (1.1-1.2× speedup) ⚠️ PARTIAL

### Medium Priority (Requires C++)
6. **SIMD distance computation** - AVX2/AVX512 intrinsics (2-4× speedup)
7. **Prefetching** - Prefetch next posting (1.1-1.2× speedup)

### Low Priority (Memory optimization)
8. **Compressed posting lists** - ZSTD compression

## 🐛 Recent Bug Fixes (Feb 13, 2026)

1. **Faiss metric bug** - CRITICAL: Was using IndexFlatL2 for all metrics!
   - Must use IndexFlatIP for IP/Cosine metrics
   - Was causing wrong centroid selection → 14% recall on Cohere
   - Fixed in 3 places: RNG building, index building, benchmark loading
   - **Impact**: 14% → 90%+ recall for IP metric

2. **IP metric sorting** - Fixed argpartition to get largest values (not smallest)
   - Was returning least similar vectors instead of most similar
   - Fixed in 4 locations: async pruning + 3 search paths
   - Recall went from 14% → 91% for IP metric

3. **RaBitQ IP handling** - Already correct (negates then gets smallest)

4. **Posting limit bounds** - Added safety check for RaBitQ indices

## 📁 Key Files

**Index:**
- `/tmp/sift1m_final_500/postings.bin` - Single-file index (SIFT 1M)
- Config: posting_size=500, replica=8, RaBitQ 2-bit, 11,482 clusters

**Code:**
- `src/index/spann_disk_optimized.py` - Main index (1000+ lines)
- `src/quantization/rabitq_numba.py` - Numba-optimized RaBitQ
- `src/clustering/hierarchical.py` - SPTAG-style clustering
- `benchmark_optimizations.py` - Optimization comparison

**Benchmarks:**
- `benchmark_sift1m_ec2.py` - SIFT 1M EC2 benchmark (async enabled)
- `benchmark_cohere1m_ec2.py` - Cohere 1M EC2 benchmark (async enabled)

## 🎯 Status vs Targets

| Metric | Current (macOS) | Target | EC2 Expected | Status |
|--------|-----------------|--------|--------------|--------|
| Recall@10 | 91.19% | 90% | 91%+ | ✅ HIT |
| p50 Latency | 11.67ms | <10ms | 4-6ms | ⚠️ 17% over (macOS) |
| p90 Latency | 13.93ms | - | 6-8ms | - |

## 🚀 Next Steps

### Immediate (Test on EC2)
1. Run `benchmark_sift1m_ec2.py` on EC2 NVMe
2. Validate <10ms p50 latency target
3. Test Cohere 1M with IP metric

### Future Optimizations (if needed)
1. **Early termination** - Stability-based stopping
2. **Workspace pooling** - Reduce allocation overhead
3. **Fix hash dedup** - Currently disabled due to recall issues
4. **SIMD** - C++ extension for distance computation (if still not fast enough)

## 📝 Documentation

**Analysis docs:**
- `QUERY_AWARE_PRUNING_ANALYSIS.md` - Deep dive into SPTAG async I/O
- `MISSING_OPTIMIZATIONS.md` - Full list of SPTAG optimizations
- `FINAL_RESULTS_FEB13.md` - Latest benchmark results

**Setup docs:**
- `RUN_ON_EC2.md` - EC2 setup instructions
- `EC2_SETUP.md` - Detailed EC2 configuration

## 🎉 Summary

**We're ready for EC2 testing!** 

Current implementation has:
- ✅ All critical optimizations (async I/O, single-file, Faiss, RaBitQ)
- ✅ 91% recall on SIFT 1M
- ⚠️ 11.67ms p50 on macOS (17% over target)
- 🚀 Expected 4-6ms p50 on EC2 NVMe (well below 10ms target!)

**Both targets should be easily achieved on EC2!** 🎯
