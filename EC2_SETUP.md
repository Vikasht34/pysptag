# EC2 Setup Guide for SIFT1M Testing

## 1. Launch EC2 Instance

**Recommended Instance:**
- Type: `t3.xlarge` (4 vCPU, 16 GB RAM) or `t3.2xlarge` (8 vCPU, 32 GB RAM)
- AMI: Amazon Linux 2023 or Ubuntu 22.04
- Storage: 30 GB EBS (gp3)

## 2. Connect to EC2

```bash
ssh -i your-key.pem ec2-user@<EC2-IP>
```

## 3. Install Dependencies

```bash
# Update system
sudo yum update -y  # Amazon Linux
# OR
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Install Python 3.11
sudo yum install -y python3.11 python3.11-pip  # Amazon Linux
# OR
sudo apt install -y python3.11 python3.11-pip  # Ubuntu

# Install required packages
python3.11 -m pip install --upgrade pip
python3.11 -m pip install numpy scipy
```

## 4. Clone Repository

```bash
cd ~
git clone https://github.com/Vikasht34/pysptag.git
cd pysptag
```

## 5. Download SIFT1M Dataset

```bash
# Create data directory
mkdir -p data

# Download SIFT1M (168 MB compressed, 577 MB extracted)
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

# Extract
tar -xzf sift.tar.gz

# Verify files
ls -lh sift/
# Should see:
# sift_base.fvecs (516 MB) - 1M vectors
# sift_query.fvecs (5.2 MB) - 10K queries
# sift_learn.fvecs (51.6 MB) - 100K training vectors
# sift_groundtruth.ivecs (4 MB) - ground truth

cd ~/pysptag
```

## 6. Update Test Script Path

```bash
# Edit test_sift1m.py to use correct paths
sed -i "s|/Users/viktari/pysptag|$HOME/pysptag|g" test_sift1m.py
```

## 7. Run SIFT1M Test

```bash
python3.11 test_sift1m.py
```

## Expected Output

```
================================================================================
SIFT1M Test - SPANN + RaBitQ + Replication
================================================================================

Loading SIFT1M dataset...
✓ Loaded in 15.23s
  Base: (1000000, 128)
  Queries: (10000, 128)
  Groundtruth: (10000, 100)

Building SPANN+RaBitQ index...
  Parameters: replica_count=8, target_posting_size=10000, bq=4

✓ Build time: 45.67s (0.8 min)
  Clusters: 100
  Avg posting size: 10000.0

Searching 10000 queries...
  Processed 1000/10000 queries...
  Processed 2000/10000 queries...
  ...

✓ Search complete: 120.45s
  QPS: 83.02

📊 Results:
  Recall@1:   85.23%
  Recall@10:  92.45%
  Recall@100: 95.67%

================================================================================
✓ SIFT1M TEST COMPLETE
================================================================================
```

## 8. Monitor Resources

In another terminal:
```bash
# Monitor CPU and memory
htop

# OR
watch -n 1 'free -h && echo && ps aux | head -20'
```

## 9. Save Results

```bash
# Run test and save output
python3.11 test_sift1m.py | tee sift1m_results.txt

# Download results to local machine
# On your local machine:
scp -i your-key.pem ec2-user@<EC2-IP>:~/pysptag/sift1m_results.txt .
```

## Troubleshooting

### Out of Memory
- Use smaller instance or reduce `target_posting_size`
- Reduce `replica_count` from 8 to 4

### Slow Performance
- Use compute-optimized instance (c6i.xlarge)
- Increase `max_check` parameter

### Dataset Download Fails
```bash
# Alternative mirror
wget http://corpus-texmex.irisa.fr/sift.tar.gz
```

## Cost Estimate

- t3.xlarge: ~$0.17/hour
- Test duration: ~5 minutes
- Total cost: < $0.02 per test run
