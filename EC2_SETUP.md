# EC2 Setup and Testing Guide

## Prerequisites

1. **EC2 Instance**: 
   - Type: `r5.2xlarge` or larger (8 vCPUs, 64GB RAM)
   - OS: Amazon Linux 2 or Ubuntu 22.04
   - Storage: 100GB+ EBS volume (gp3 recommended)

2. **Python Environment**:
   ```bash
   sudo yum install -y python3 python3-pip git  # Amazon Linux
   # OR
   sudo apt-get update && sudo apt-get install -y python3 python3-pip git  # Ubuntu
   ```

3. **Mount EBS Volume** (if using separate volume):
   ```bash
   sudo mkfs -t ext4 /dev/nvme1n1  # Format (first time only)
   sudo mkdir -p /mnt/ebs
   sudo mount /dev/nvme1n1 /mnt/ebs
   sudo chown -R ec2-user:ec2-user /mnt/ebs  # Amazon Linux
   # OR
   sudo chown -R ubuntu:ubuntu /mnt/ebs  # Ubuntu
   ```

## Setup

1. **Clone Repository**:
   ```bash
   cd ~
   git clone https://github.com/Vikasht34/pysptag.git
   cd pysptag
   ```

2. **Install Dependencies**:
   ```bash
   pip3 install numpy numba --user
   ```

3. **Download SIFT1M Dataset**:
   ```bash
   mkdir -p data/sift
   cd data/sift
   
   # Download SIFT1M
   wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
   tar -xzf sift.tar.gz
   
   cd ~/pysptag
   ```

## Run Tests

### 1. Memory-Based SPANN (Fast, In-Memory)
```bash
cd ~/pysptag
python3 test_sift1m_efficient.py
```

**Expected Results**:
- 4-bit: ~4ms p50, ~250 QPS, 92% recall
- 2-bit: ~4ms p50, ~220 QPS, 92% recall
- Build time: ~2-3 minutes

### 2. Disk-Based SPANN (Billion-Scale Ready)
```bash
cd ~/pysptag
python3 test_ec2_disk.py
```

**Expected Results**:
- 4-bit: ~5-10ms p50, ~150 QPS, 92% recall
- Disk usage: ~750MB (compressed from 3GB)
- Build time: ~3-5 minutes

**Note**: First run builds the index, subsequent runs load from disk.

## Configuration Options

Edit `test_ec2_disk.py` to customize:

```python
# Line 60: Change EBS mount point
INDEX_DIR = '/mnt/ebs/spann_index'  # Your EBS path

# Line 85: Use BKTree instead of KDTree
tree_type='BKT',  # Slower but more accurate

# Line 84: Adjust quantization
bq=2,  # 1-bit, 2-bit, or 4-bit

# Line 83: Disable quantization
use_rabitq=False,  # No compression, slower but highest recall
```

## Performance Tuning

### For Faster Search:
```python
search_internal_result_num=64,  # Fewer centroids (faster, lower recall)
max_check=2048,  # Fewer candidates (faster, lower recall)
```

### For Higher Recall:
```python
search_internal_result_num=256,  # More centroids (slower, higher recall)
max_check=8192,  # More candidates (slower, higher recall)
```

## Monitoring

### Check Disk Usage:
```bash
du -sh /mnt/ebs/spann_index_*
```

### Monitor Memory:
```bash
watch -n 1 free -h
```

### Monitor CPU:
```bash
htop
```

## Troubleshooting

### Out of Memory:
- Use smaller `replica_count` (e.g., 4 instead of 6)
- Use disk-based instead of memory-based
- Increase EC2 instance size

### Slow Search:
- Use KDTree instead of BKTree (`tree_type='KDT'`)
- Reduce `search_internal_result_num`
- Use 2-bit or 4-bit quantization

### Low Recall:
- Increase `search_internal_result_num`
- Increase `max_check`
- Use 4-bit quantization or no-quant

## Next Steps: Billion-Scale

Once SIFT1M works, scale to billion vectors:

1. **Larger Dataset**: Download DEEP1B or SIFT1B
2. **Larger Instance**: Use `r5.8xlarge` (32 vCPUs, 256GB RAM)
3. **Larger EBS**: 1TB+ gp3 volume
4. **Adjust Parameters**:
   ```python
   target_posting_size=10000,  # Larger posting lists
   replica_count=8,  # More replicas for recall
   ```

## Results to Expect

### SIFT1M (1M vectors):
- **Memory-based**: 3-4ms latency, 92% recall
- **Disk-based**: 5-10ms latency, 92% recall
- **Disk usage**: ~750MB (4-bit quantization)

### Billion-scale (1B vectors):
- **Disk-based**: 10-20ms latency, 90%+ recall
- **Disk usage**: ~750GB (4-bit quantization)
- **RAM usage**: ~10GB (only centroids in memory)
