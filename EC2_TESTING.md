# Running Tests on EC2

## Quick Start

```bash
cd ~/pysptag
git pull

# SIFT1M (L2 metric, 128D)
python3 test_sift1m_efficient.py

# Cohere 1M (IP metric, 768D)
python3 test_cohere_1m_efficient.py
```

## Expected Results

### SIFT1M (1M vectors, 128D, L2)
- **1-bit**: 79-85% recall, ~5ms latency
- **2-bit**: 90-93% recall, ~9ms latency ✓
- **4-bit**: 92-93% recall, ~9ms latency ✓
- **no-quant**: 92-93% recall, ~6ms latency

### Cohere 1M (1M vectors, 768D, IP)
- **1-bit**: 75-85% recall, ~10ms latency
- **2-bit**: 85-95% recall, ~15ms latency ✓
- **4-bit**: 90-95% recall, ~15ms latency ✓
- **no-quant**: 95%+ recall, ~10ms latency

## Data Location

- SIFT1M: `~/pysptag/data/sift/`
- Cohere 1M: `~/pysptag/data/documents-1m.hdf5`

## Test Features

Both tests use **shared clustering** for efficiency:
- Cluster once (~60-120s)
- Quantize 4 times (~20-40s total)
- **3× faster** than separate builds

## Output

Each test produces:
- Per-config results (recall, latency, QPS)
- Summary table comparing all configs
- Time savings from shared clustering
- Recommendations for production use

## Troubleshooting

**If HDF5 structure error:**
```python
# Check HDF5 structure
import h5py
with h5py.File('~/pysptag/data/documents-1m.hdf5', 'r') as f:
    print(list(f.keys()))
    for key in f.keys():
        print(f"{key}: {f[key].shape if hasattr(f[key], 'shape') else 'Group'}")
```

**If out of memory:**
- Reduce `replica_count` (default: 6)
- Reduce `target_posting_size` (default: 5000)
- Use smaller test subset

## Performance Tips

For faster testing:
- Use fewer queries (change `[:100]` to `[:10]`)
- Reduce `max_check` parameter
- Test only specific bit levels

For production benchmarking:
- Use all queries (1000+)
- Average over multiple runs
- Use fixed random seed for reproducibility
