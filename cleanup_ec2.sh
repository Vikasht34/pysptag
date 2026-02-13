#!/bin/bash
# Cleanup script for EC2 - removes large test files and build artifacts

echo "Cleaning up EC2 disk space..."

# Remove Python cache
echo "Removing Python cache..."
find ~/pysptag -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find ~/pysptag -type f -name "*.pyc" -delete 2>/dev/null

# Remove test output files
echo "Removing test outputs..."
rm -f ~/pysptag/*.log 2>/dev/null
rm -f ~/pysptag/build.log 2>/dev/null

# Remove any pickle/numpy saved files
echo "Removing saved index files..."
rm -f ~/pysptag/*.pkl 2>/dev/null
rm -f ~/pysptag/*.npy 2>/dev/null
rm -f ~/pysptag/*.npz 2>/dev/null

# Show disk usage
echo ""
echo "Disk usage after cleanup:"
du -sh ~/pysptag
du -sh ~/pysptag/data 2>/dev/null || echo "No data directory"

echo ""
echo "Cleanup complete!"
echo ""
echo "To free more space, you can remove datasets:"
echo "  rm -rf ~/pysptag/data/sift      # SIFT1M dataset"
echo "  rm -rf ~/pysptag/data/*.hdf5    # Cohere dataset"
