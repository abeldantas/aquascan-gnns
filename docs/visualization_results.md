# Visualization Analysis Results

## Key Findings

After analyzing the simulation with trajectory tracking at 5-tick intervals (x128 speed), we've determined the optimal configuration for data generation:

### Optimal Parameters
- **Tick Interval**: 5 ticks (640 seconds of simulated time)
- **Time Horizon**: 235 ticks total (~8.4 hours simulated)
- **Snapshots**: 48 frames per simulation run
- **Speed**: x128 (1 tick = 128 seconds)

### Why This Configuration Works

1. **Smooth Motion Tracking**: 5-tick intervals capture entity movement without redundancy
2. **Reasonable File Size**: ~2MB per simulation run (48 snapshots)
3. **Sufficient Coverage**: 8+ hours captures multiple detection events and movement patterns
4. **Prediction Challenge**: At this framerate, 30-tick predictions (6 snapshots) are reasonable, while 300-tick predictions (60 snapshots) are genuinely challenging

### Visualization Grid

The 48-frame grid (6×8 layout) with trajectory lines provides clear visibility of:
- Entity movement patterns (Brownian vs sinusoidal)
- Detection event clustering
- Network topology stability
- Ocean current effects on sensor drift

### Reference Visualization

See `docs/reference_48frame_grid.png` for the 48-frame trajectory visualization that informed these decisions.

The grid shows 48 snapshots (6×8 layout) captured at 5-tick intervals, with:
- Trajectory lines for each marine entity
- Color-coded entities maintaining consistent colors
- Clear detection events (bright green borders)
- Network topology evolution
- ~8.4 hours of simulated time

## Data Size Estimates

Based on the 5-tick interval configuration:
- **Per snapshot**: ~40KB (581 nodes + ~1200 edges)
- **Per run**: ~2MB (48 snapshots)
- **Target**: 1000-2000 runs for comprehensive training data
- **Total raw data**: 2-4GB (leaves room for graph processing overhead)

This configuration balances data quality with computational efficiency.

## Speed Verification

The batch generation uses **exactly x128 speed** as validated in the visualizations:
- Each tick represents 128 seconds of simulated time
- This is set in `aquascan/run_simulation.py` before any simulation executes
- Verified with `scripts/verify_speed_consistency.py` showing identical outputs
