#!/bin/bash
# EC2 Setup Script for SPTAG SPANN Testing

set -e

echo "=========================================="
echo "EC2 Setup for SPTAG SPANN"
echo "=========================================="

# Check if NVMe is mounted
if [ ! -d "/mnt/nvme" ]; then
    echo "⚠️  /mnt/nvme not found. Creating directory..."
    sudo mkdir -p /mnt/nvme
    sudo chown $USER:$USER /mnt/nvme
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install numpy numba faiss-cpu h5py --quiet

# Check data file
DATA_FILE="$HOME/pysptag/cohere-wikipedia-768-angular.hdf5"
if [ ! -f "$DATA_FILE" ]; then
    echo "⚠️  Data file not found: $DATA_FILE"
    echo "Please download Cohere 1M dataset first"
    exit 1
fi

echo "✓ Data file found: $DATA_FILE"

# Run test
echo ""
echo "=========================================="
echo "Running optimized test..."
echo "=========================================="
cd ~/pysptag
python3 test_ec2_cohere_optimized.py

echo ""
echo "✓ Test complete!"
