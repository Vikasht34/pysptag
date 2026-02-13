# Quick Start: Run on EC2

## 1. Setup
```bash
# Install dependencies
pip3 install numpy h5py numba

# Clone repo
git clone https://github.com/Vikasht34/pysptag.git
cd pysptag
```

## 2. Download Datasets
```bash
# SIFT1M (128D, L2)
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5

# Cohere 1M (768D, IP)
wget http://ann-benchmarks.com/cohere-wiki-1m-768-ip.hdf5
```

## 3. Run Tests

### SIFT1M (Disk-Based)
```bash
python3 test_sift1m_disk.py
```

### Cohere 1M (Disk-Based)
```bash
python3 test_cohere_1m_disk.py
```

## Expected Results

### SIFT1M
- Build: ~80-100s (first run only)
- Search: 3-5ms p50 latency
- Recall: 80-93%
- Disk: ~50-200 MB

### Cohere 1M
- Build: ~120-150s (first run only)
- Search: 6-10ms p50 latency
- Recall: 70-90%
- Disk: ~300-1200 MB

## Notes
- First run builds index and saves to disk
- Subsequent runs load from disk (~1-2s)
- All posting lists stored on EBS
- Only centroids kept in RAM
