# Aquascan Scripts

This directory contains utility scripts for the Aquascan project.

## Visualization Scripts

### `generate_snapshots.py`
Generates PNG visualizations from headless simulations with trajectory tracking.
- Uses x128 speed (1 tick = 128 seconds)
- Creates snapshots at configurable intervals
- Shows entity trajectories with unique colors

### `visualize_snapshots.py`
Combines multiple snapshots into grids or animations.
- Grid view for easy comparison
- Animated GIF generation

### `visualization_examples.py`
Interactive demo of various visualization scenarios.

## Model Training Scripts

### `kalman_eval.py`
Evaluates the Kalman filter baseline on test data.

### `gnn_train.py`
Trains the Graph Neural Network model.

### `gnn_eval.py`
Evaluates trained GNN models.

## Verification Scripts

### `verify_speed_consistency.py`
Verifies that batch generation uses the same x128 speed as visualizations.
- Runs identical simulations via both approaches
- Compares positions, timing, and statistics
- Confirms data generation consistency

Run to verify:
```bash
python scripts/verify_speed_consistency.py
```

## Quick Tests

### `test_snapshots.sh`
Quick 2-minute visualization test.

### `test_batch_generation.sh`
Test batch generation with 5 runs.
