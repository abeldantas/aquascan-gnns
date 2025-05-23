#!/bin/bash
# test_snapshots.sh - Quick visualization test script
#
# This script runs a short 2-minute simulation and generates visualization
# snapshots every 30 seconds. Used for quickly verifying the visualization
# system is working correctly.
#
# Usage: ./test_snapshots.sh

echo "Testing snapshot generation with a short simulation..."

# Generate snapshots for a 2-minute simulation (120 seconds = 120 ticks at x128 speed)
# Capture every 30 seconds (30 ticks)
python scripts/generate_snapshots.py \
    --seed 42 \
    --ticks 120 \
    --interval 30 \
    --output visualizations

echo "Done! Check the visualizations/ directory for output."
