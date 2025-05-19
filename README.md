# Aquascan: A Digital Twin Architecture for Marine Motion Modeling

## Overview
Aquascan is a simulation environment for modeling spatiotemporal behavior of marine life via a distributed ocean twin simulation. The system simulates a network of IoT devices (ε-nodes) in the water, relay nodes (σ-nodes), and marine entities (θ-contacts).

## Current Status
- ✅ Project initialized
- ✅ Basic structure created
- ⬜ Configuration module
- ⬜ Hexagonal grid utilities
- ⬜ Entity classes (ε-nodes, σ-nodes, θ-contacts)
- ⬜ Ocean current simulation
- ⬜ Simulation loop
- ⬜ Visualization
- ⬜ Documentation

## Project Structure
```
aquascan-gnns/
├── config/            # Simulation configuration parameters
├── simulation/        # Core simulation components
│   ├── ocean_area.py  # Defines hex grid, ε-/σ-node positioning
│   ├── entities.py    # θ-contact behavior
│   ├── sensors.py     # Signal simulation, detection
│   ├── communication.py  # ε-ε link logic
│   └── simulation_loop.py  # Master tick loop
├── visual/
│   └── bokeh_app.py   # Bokeh layout, callbacks, real-time display
└── utils/
    └── hex_grid.py    # Coordinate math for hexagonal grid
```

## Setup
1. Clone the repository
2. Activate the virtual environment:
   ```
   source venv/bin/activate  # On Unix/macOS
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
(To be updated as implementation progresses)

## Parameters
- Simulation area: 30km × 16km rectangular offshore strip
- Resolution: 1km spacing (570 ε-nodes)
- Time step: 1 second
- Simulation duration: Configurable, default equivalent to 1 week

## Marine Entities
- European seabass (Dicentrarchus labrax)
- Atlantic horse mackerel (Trachurus trachurus)
- Bottlenose dolphin (Tursiops truncatus)

## Future Work
- Graph Neural Network (GNN) implementation for prediction
- Comparison with Kalman filter baseline
