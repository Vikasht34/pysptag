# Disk-Based SPANN Testing on EC2

## Overview

Disk-based SPANN for billion-scale datasets. Posting lists are saved to disk and loaded on-demand during search.

## Setup on EC2

### 1. Install Dependencies
```bash
pip3 install numpy scikit-learn h5py numba
```

### 2. Clone Repository
```bash
git clone https://github.com/Vikasht34/pysptag.git
cd pysptag
git pull  # Get latest changes
```

### 3. Download Datasets

**SIFT1M** (128D, L2 metric):
```bash
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
```

**Cohere 1M** (768D, IP metric):
```bash
wget http://ann-benchmarks.com/cohere-wiki-1m-768-ip.hdf5
```

## Running Tests

### SIFT1M Test
```bash
python3 test_sift1m_disk.py
```

**Expected output**:
- Build time: ~80-100s (first run only)
- Search: 100 queries
- Latency: 3-5ms p50
- Recall: 80-93% depending on quantization
- Disk usage: ~50-200 MB per index

### Cohere 1M Test
```bash
python3 test_cohere_1m_disk.py
```

**Expected output**:
- Build time: ~120-150s (first run only)
- Search: 100 queries
- Latency: 5-10ms p50 (768D is larger)
- Recall: 70-90% depending on quantization
- Disk usage: ~300-1200 MB per index

## How It Works

### Build Phase
1. **Clustering**: K-means to create centroids
2. **Replication**: Assign each vector to 6 nearest centroids
3. **Quantization**: RaBitQ quantization (1-bit, 2-bit, or 4-bit)
4. **Save to disk**: Each posting list saved as separate file

### Search Phase
1. **Find nearest centroids** (in-memory, fast)
2. **Load posting lists from disk** (on-demand)
3. **Search with RaBitQ** (quantized distance estimation)
4. **Rerank with true distances** (final top-k)

## Disk Structure

```
./sift1m_index_2bit/
├── metadata.pkl          # Centroids, BKTree, RNG
└── postings/
    ├── posting_0.pkl     # Posting list 0
    ├── posting_1.pkl     # Posting list 1
    └── ...
```

Each posting file contains:
- `posting_ids`: Vector IDs in this posting
- `codes`: Quantized codes
- `rabitq`: RaBitQ quantizer object

## Performance Tuning

### For Lower Latency
```python
# Reduce centroids checked
search_internal_result_num=64  # Default: 128

# Reduce candidates
max_check=2000  # Default: 4000

# Use 1-bit quantization
bq=1  # Faster but lower recall
```

### For Higher Recall
```python
# Check more centroids
search_internal_result_num=256

# Check more candidates
max_check=8000

# Use 4-bit quantization
bq=4  # Slower but higher recall
```

### For Lower Disk Usage
```python
# Use 1-bit quantization
bq=1  # 32× compression

# Larger posting lists
target_posting_size=10000  # Fewer postings
```

## Monitoring

### Check Disk Usage
```bash
du -sh ./sift1m_index_*
du -sh ./cohere1m_index_*
```

### Check I/O Performance
```bash
# Monitor disk I/O during search
iostat -x 1
```

### Check Memory Usage
```bash
# Memory should stay low (only centroids in RAM)
free -h
```

## Expected Results

### SIFT1M (1M vectors, 128D, L2)
| Config | Latency p50 | Recall@10 | Disk Size |
|--------|-------------|-----------|-----------|
| 1-bit  | 3-4ms | 80-85% | ~50 MB |
| 2-bit  | 4-5ms | 90-93% | ~100 MB |
| 4-bit  | 4-5ms | 92-94% | ~200 MB |

### Cohere 1M (1M vectors, 768D, IP)
| Config | Latency p50 | Recall@10 | Disk Size |
|--------|-------------|-----------|-----------|
| 1-bit  | 6-8ms | 70-80% | ~300 MB |
| 2-bit  | 8-10ms | 85-90% | ~600 MB |
| 4-bit  | 8-10ms | 88-92% | ~1200 MB |

## Troubleshooting

### Out of Memory
- Reduce `replica_count` (default: 6)
- Increase `target_posting_size` (fewer clusters)

### Slow Search
- Check EBS IOPS (should be >3000)
- Use gp3 volumes with provisioned IOPS
- Reduce `search_internal_result_num`

### Low Recall
- Increase `search_internal_result_num`
- Increase `max_check`
- Use higher bit quantization (4-bit)

## Scaling to Billion-Scale

For 1B+ vectors:
1. Use larger `target_posting_size` (10K-50K)
2. Use 1-bit or 2-bit quantization
3. Use EBS with high IOPS (10K+)
4. Consider sharding across multiple machines

## Notes

- **First run**: Builds index and saves to disk (~2-3 minutes)
- **Subsequent runs**: Loads from disk (~1-2 seconds)
- **Disk I/O**: Main bottleneck for search latency
- **EBS performance**: Use gp3 with provisioned IOPS for best results
