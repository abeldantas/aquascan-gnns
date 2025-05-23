#!/bin/bash
# test_batch_generation.sh - Test script for optimized batch data generation
#
# This script runs a small test of the batch generation system using the
# optimized 5-tick interval configuration determined through visualization analysis.
# It generates 5 test simulations to verify the system works before running
# the full dataset generation.
#
# Usage: ./test_batch_generation.sh

echo "🧪 Testing optimized batch generation (5-tick intervals)..."
echo ""

# Create test output directory
mkdir -p data/test_5tick

# Run just 5 simulations to test
echo "Running 5 test simulations..."
python -m aquascan.batch.generate \
    --cfg configs/optimal_5tick.yml \
    --runs 5 \
    --out data/test_5tick \
    --jobs 4 \
    --validate

echo ""
echo "✅ Test complete! Check data/test_5tick/ for output files."
echo ""
echo "To run full dataset generation (1000 runs):"
echo "python -m aquascan.batch.generate --cfg configs/optimal_5tick.yml --runs 1000 --out data/raw_5tick --jobs 8"
