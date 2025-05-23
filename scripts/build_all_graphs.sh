#!/bin/bash
# Build graph datasets from the blessed 5-tick interval data
# 
# 🚨 UPDATE: Now uses PARALLEL processing by default!
# Old sequential version took 10-12 HOURS. This takes ~30 minutes.
# 
echo "🚀 Starting PARALLEL graph building..."
echo "   (Old version took 10-12 hours, this takes ~30 min)"
echo ""

# Just run the parallel version
exec ./scripts/build_all_graphs_parallel.sh
