#!/bin/bash
# Build graph datasets from the blessed 5-tick interval data
# 
# Our data has 235 ticks total (48 snapshots), so we adjust horizons accordingly:
# - 30 ticks = 64 seconds (easy)
# - 100 ticks = 3.5 minutes (moderate) 
# - 150 ticks = 5.3 minutes (challenging)

echo "🔥 Building graph datasets from the blessed drop..."
echo ""

# Check if raw data exists
if [ ! -d "data/raw_5tick" ]; then
    echo "❌ Error: data/raw_5tick not found!"
    echo "   Run: python -m aquascan.batch.generate --cfg configs/optimal_5tick.yml --runs 1000"
    exit 1
fi

echo "📊 Found raw data! Building graphs..."

# Easy task - 30 tick horizon (64 seconds)
echo ""
echo "1️⃣ Building 30-tick horizon (easy - 64 seconds)..."
python -m aquascan.dataset.build_graphs \
    --raw data/raw_5tick \
    --out data/processed_30tick \
    --context 60 \
    --horizon 30 \
    --split 0.7 0.15 0.15

# Moderate task - 100 tick horizon (3.5 minutes)
echo ""
echo "2️⃣ Building 100-tick horizon (moderate - 3.5 minutes)..."
python -m aquascan.dataset.build_graphs \
    --raw data/raw_5tick \
    --out data/processed_100tick \
    --context 60 \
    --horizon 100 \
    --split 0.7 0.15 0.15

# Challenging task - 150 tick horizon (5.3 minutes)  
echo ""
echo "3️⃣ Building 150-tick horizon (challenging - 5.3 minutes)..."
python -m aquascan.dataset.build_graphs \
    --raw data/raw_5tick \
    --out data/processed_150tick \
    --context 60 \
    --horizon 150 \
    --split 0.7 0.15 0.15

echo ""
echo "✅ Graph datasets ready! Size check:"
du -sh data/processed_*

echo ""
echo "🚀 Next: Upload to Colab and let the model feast!"
