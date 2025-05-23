# Visualization Guide for Aquascan Simulations

## Overview

This guide explains how to use the visualization tools to verify that the Aquascan simulation is generating sensible data before running experiments.

## Quick Start

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Run a quick test
./test_snapshots.sh

# 3. Check the output
ls -la visualizations/
```

## Tools Available

### 1. `generate_snapshots.py`
Main tool for creating visual snapshots from headless simulations.

**Key Features:**
- Runs simulation at x128 speed by default
- Captures PNG snapshots at specified intervals
- Shows all entities, connections, and detections
- Includes simulation statistics and current indicators

**Usage:**
```bash
python scripts/generate_snapshots.py [options]

Options:
  --seed SEED        Random seed (default: 42)
  --ticks TICKS      Total simulation ticks (default: 600)
  --interval INT     Snapshot interval in ticks (default: 60)
  --output DIR       Output directory (default: visualizations)
```

### 2. `visualize_snapshots.py`
Combines multiple snapshots into grids or animations.

**Usage:**
```bash
# Create a grid view
python scripts/visualize_snapshots.py <snapshot_dir> --grid --cols 3

# Create an animated GIF
python scripts/visualize_snapshots.py <snapshot_dir> --animate --fps 2
```

### 3. `visualization_examples.py`
Interactive script with various example scenarios.

```bash
python scripts/visualization_examples.py
```

## What to Look For

When reviewing the generated visualizations, check for:

### 1. **Entity Movement Patterns**
- Fish (Brownian motion) should show random walks with occasional direction changes
- Dolphins (sinusoidal motion) should show wave-like patterns
- Entities should generally stay near the deployment area but can migrate outside

### 2. **Detection Events**
- Green entities = currently detected by at least one sensor
- Gray entities = not detected
- Purple lines = active detection connections
- Detection radius is 200m (0.2km)

### 3. **Network Topology**
- Blue solid lines = permanent connections (<5km)
- Blue dashed lines = intermittent connections (5-10km)
- Each node should have 3-5 connections
- Network should remain connected

### 4. **Ocean Currents**
- Arrow shows current direction and strength
- Epsilon nodes should drift with currents
- Current patterns should be consistent but vary spatially

## Recommended Workflow

1. **Initial Check** - Run a 2-minute simulation to verify basics:
   ```bash
   python scripts/generate_snapshots.py --ticks 120 --interval 30
   ```

2. **Detailed Analysis** - Run longer simulation with more snapshots:
   ```bash
   python scripts/generate_snapshots.py --ticks 600 --interval 20
   ```

3. **Create Summary** - Generate grid and animation:
   ```bash
   LATEST_DIR=$(ls -td visualizations/run_* | head -1)
   python scripts/visualize_snapshots.py $LATEST_DIR --grid
   python scripts/visualize_snapshots.py $LATEST_DIR --animate
   ```

4. **Compare Seeds** - Check variability across runs:
   ```bash
   for seed in 1 2 3; do
     python scripts/generate_snapshots.py --seed $seed --ticks 120 --interval 60
   done
   ```

## Interpreting Results

### Good Signs ✅
- Smooth entity movements without teleportation
- Detection events occur when entities are visually close to sensors
- Network maintains connectivity over time
- Entities exhibit expected motion patterns

### Potential Issues ❌
- Entities jumping large distances between frames
- Detections occurring at unrealistic distances
- Network becoming disconnected
- All entities moving in exactly the same direction

## Performance Notes

- Generating snapshots is CPU-intensive (matplotlib rendering)
- Each snapshot takes ~1-2 seconds to generate
- For long simulations, use larger intervals (e.g., every 60-120 ticks)
- Animations work best with 10-20 frames total

## Next Steps

Once you've verified the simulation is working correctly:

1. Generate larger datasets for training:
   ```bash
   python -m aquascan.batch.generate --runs 100 --out data/raw --jobs 8
   ```

2. Build graph datasets with different horizons:
   ```bash
   python -m aquascan.dataset.build_graphs --horizon 30   # Original
   python -m aquascan.dataset.build_graphs --horizon 300  # Challenging
   ```

3. Compare model performance across different prediction horizons

## Troubleshooting

**No snapshots generated:**
- Check that matplotlib is installed: `pip install matplotlib pillow`
- Verify the simulation is running: check for error messages

**Snapshots look wrong:**
- Verify coordinate system (should show 30x16km area)
- Check detection radius setting (should be 0.2km)
- Ensure random seed is set for reproducibility

**Memory issues:**
- Reduce number of snapshots with larger --interval
- Generate snapshots in batches for very long simulations
