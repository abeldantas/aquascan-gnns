#!/usr/bin/env python3
"""
Build graph datasets from the optimized 5-tick interval raw data.

Based on our visualization-validated dataset:
- Total ticks: 235 (48 snapshots)
- Snapshot interval: 5 ticks
- Each tick = 128 seconds at x128 speed

This script shows how to build graphs with different prediction horizons.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Building Graph Datasets from Raw HDF5 Files")
    print("=" * 60)
    
    # Check if raw data exists
    if not os.path.exists("data/raw_5tick"):
        print("❌ Error: data/raw_5tick not found!")
        print("   Run batch generation first:")
        print("   python -m aquascan.batch.generate --cfg configs/optimal_5tick.yml --runs 1000")
        sys.exit(1)
    
    print("📊 Dataset Info:")
    print("   - Raw data: 235 ticks per simulation")
    print("   - Snapshots: 48 (every 5 ticks)")
    print("   - Speed: x128 (1 tick = 128 seconds)")
    print("")
    
    # Build 30-tick horizon dataset (EASY - 30 seconds prediction)
    print("📦 Building 30-tick horizon dataset (Easy task)...")
    print("   - Context: 60 ticks (12 snapshots)")
    print("   - Horizon: 30 ticks (6 snapshots) = 64 seconds ahead")
    print("   - Max graph time: tick 175 (235 - 60)")
    
    cmd_30 = [
        "python", "-m", "aquascan.dataset.build_graphs",
        "--raw", "data/raw_5tick",
        "--out", "data/processed_30tick", 
        "--context", "60",
        "--horizon", "30",
        "--split", "0.7", "0.15", "0.15"
    ]
    
    result = subprocess.run(cmd_30)
    if result.returncode != 0:
        print("❌ Failed to build 30-tick dataset")
        sys.exit(1)
    
    print("\n✅ 30-tick dataset complete!")
    print("-" * 60)
    
    # Build 150-tick horizon dataset (CHALLENGING - 3.2 minutes prediction)
    print("\n📦 Building 150-tick horizon dataset (Challenging task)...")
    print("   - Context: 60 ticks (12 snapshots)")
    print("   - Horizon: 150 ticks (30 snapshots) = 320 seconds ahead")
    print("   - Max graph time: tick 25 (only ~53% of simulation usable)")
    
    cmd_150 = [
        "python", "-m", "aquascan.dataset.build_graphs",
        "--raw", "data/raw_5tick",
        "--out", "data/processed_150tick",
        "--context", "60", 
        "--horizon", "150",
        "--split", "0.7", "0.15", "0.15"
    ]
    
    result = subprocess.run(cmd_150)
    if result.returncode != 0:
        print("❌ Failed to build 150-tick dataset")
        sys.exit(1)
    
    print("\n✅ 150-tick dataset complete!")
    print("-" * 60)
    
    # Alternative: Build 100-tick horizon (MODERATE - 2.1 minutes)
    print("\n📦 Building 100-tick horizon dataset (Moderate task)...")
    print("   - Context: 60 ticks (12 snapshots)")
    print("   - Horizon: 100 ticks (20 snapshots) = 213 seconds ahead")
    print("   - Max graph time: tick 75 (more data available)")
    
    cmd_100 = [
        "python", "-m", "aquascan.dataset.build_graphs",
        "--raw", "data/raw_5tick",
        "--out", "data/processed_100tick",
        "--context", "60",
        "--horizon", "100", 
        "--split", "0.7", "0.15", "0.15"
    ]
    
    result = subprocess.run(cmd_100)
    if result.returncode != 0:
        print("❌ Failed to build 100-tick dataset")
        sys.exit(1)
    
    print("\n✅ All datasets built successfully!")
    print("\n📊 Summary:")
    print("   - 30-tick: Easy baseline (64 seconds prediction)")
    print("   - 100-tick: Moderate challenge (3.5 minutes prediction)")
    print("   - 150-tick: Hard challenge (5.3 minutes prediction)")
    print("\n🎯 Next: Upload to Colab and train models!")

if __name__ == "__main__":
    main()
