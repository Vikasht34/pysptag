#!/bin/bash
# Quick EC2 setup script - run this after connecting to EC2

set -e

echo "=========================================="
echo "PySPTAG EC2 Quick Setup"
echo "=========================================="

# Install Python
echo "Installing Python 3.11..."
sudo yum install -y python3.11 python3.11-pip || sudo apt install -y python3.11 python3.11-pip

# Install dependencies
echo "Installing dependencies..."
python3.11 -m pip install --upgrade pip
python3.11 -m pip install numpy scipy

# Clone repo
echo "Cloning repository..."
cd ~
git clone https://github.com/Vikasht34/pysptag.git
cd pysptag

# Download SIFT1M
echo "Downloading SIFT1M dataset..."
mkdir -p data
cd data
wget -q --show-progress ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
cd ..

# Fix paths in test script
echo "Updating paths..."
sed -i "s|/Users/viktari/pysptag|$HOME/pysptag|g" test_sift1m.py

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Run test with:"
echo "  cd ~/pysptag"
echo "  python3.11 test_sift1m.py"
