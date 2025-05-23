#!/bin/bash
# Parallelized graph building with proper progress tracking
#
# Uses all CPU cores and shows real-time progress for each horizon

echo "🚀 Parallel Graph Builder - Let's cook with gas!"
echo ""

# Check if raw data exists
if [ ! -d "data/raw_5tick" ]; then
    echo "❌ Error: data/raw_5tick not found!"
    echo "   Run: python -m aquascan.batch.generate --cfg configs/optimal_5tick.yml --runs 1000"
    exit 1
fi

# Count files
FILE_COUNT=$(ls -1 data/raw_5tick/*.h5 2>/dev/null | wc -l)
echo "📊 Found ${FILE_COUNT} HDF5 files to process"

# Detect CPU cores
CORES=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())")
echo "⚡ Using ${CORES} CPU cores for parallel processing"
echo ""

# Kill the old slow process if it's running
OLD_PID=$(pgrep -f "aquascan.dataset.build_graphs")
if [ ! -z "$OLD_PID" ]; then
    echo "🔪 Killing old slow sequential process (PID: $OLD_PID)..."
    kill $OLD_PID
    sleep 2
fi

# Track total time
START_TIME=$(date +%s)

# Easy task - 30 tick horizon (64 seconds)
echo "═══════════════════════════════════════════════════════"
echo "1️⃣  HORIZON: 30 ticks (64 seconds) - EASY"
echo "═══════════════════════════════════════════════════════"
python scripts/build_graphs_parallel.py \
    --raw data/raw_5tick \
    --out data/processed_30tick \
    --context 60 \
    --horizon 30 \
    --split 0.7 0.15 0.15 \
    --jobs $CORES

if [ $? -eq 0 ]; then
    echo "✅ 30-tick dataset complete!"
else
    echo "❌ 30-tick dataset failed!"
    exit 1
fi

# Moderate task - 100 tick horizon (3.5 minutes)
echo ""
echo "═══════════════════════════════════════════════════════"
echo "2️⃣  HORIZON: 100 ticks (3.5 minutes) - MODERATE"
echo "═══════════════════════════════════════════════════════"
python scripts/build_graphs_parallel.py \
    --raw data/raw_5tick \
    --out data/processed_100tick \
    --context 60 \
    --horizon 100 \
    --split 0.7 0.15 0.15 \
    --jobs $CORES

if [ $? -eq 0 ]; then
    echo "✅ 100-tick dataset complete!"
else
    echo "❌ 100-tick dataset failed!"
    exit 1
fi

# Challenging task - 150 tick horizon (5.3 minutes)
echo ""
echo "═══════════════════════════════════════════════════════"
echo "3️⃣  HORIZON: 150 ticks (5.3 minutes) - CHALLENGING"
echo "═══════════════════════════════════════════════════════"
python scripts/build_graphs_parallel.py \
    --raw data/raw_5tick \
    --out data/processed_150tick \
    --context 60 \
    --horizon 150 \
    --split 0.7 0.15 0.15 \
    --jobs $CORES

if [ $? -eq 0 ]; then
    echo "✅ 150-tick dataset complete!"
else
    echo "❌ 150-tick dataset failed!"
    exit 1
fi

# Summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))

echo ""
echo "═══════════════════════════════════════════════════════"
echo "📊 FINAL SUMMARY"
echo "═══════════════════════════════════════════════════════"
echo "✅ All graph datasets ready!"
echo "⏱️  Total time: ${TOTAL_MINUTES} minutes (${TOTAL_TIME} seconds)"
echo ""
echo "📁 Size check:"
du -sh data/processed_*tick
echo ""
echo "🚀 Next: Upload to Colab and unleash the GNN!"
