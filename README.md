# Aquascan: A Digital Twin Architecture for Marine Motion Modeling

## Overview
Aquascan is a simulation environment for modeling spatiotemporal behavior of marine life via a distributed ocean twin simulation. The system simulates a network of IoT devices (ε-nodes) in the water, relay nodes (σ-nodes), and marine entities (θ-contacts).

The simulation focuses on a marine monitoring network deployed in a rectangular offshore strip measuring 30km along the coastline and extending 16km seaward, from 6km to 22km offshore. This area is designed to monitor marine wildlife corridors and fishing vessel activity.

## Current Status
- ✅ Project structure and core architecture
- ✅ Hexagonal grid node deployment
- ✅ Ocean current simulation with Perlin noise
- ✅ Marine entity motion models (Brownian and sinusoidal)
- ✅ Detection and communication protocols
- ✅ Real-time visualization with Bokeh
- ✅ Detection visualization (color change for detected entities)
- ⬜ Data persistence (HDF5)
- ⬜ GNN implementation for prediction
- ⬜ Comparison with Kalman filter baseline

## Project Structure
```
aquascan-gnns/
├── config/            # Simulation configuration parameters
│   └── simulation_config.py  # Central configuration file
├── simulation/        # Core simulation components
│   ├── ocean_area.py  # Deployment area and node positioning
│   ├── entities.py    # ε-nodes, σ-nodes, and θ-contact classes
│   ├── sensors.py     # Signal simulation and detection
│   ├── communication.py  # Communication protocols (RPR and SCV)
│   ├── simulation_loop.py  # Main simulation tick procedure
│   └── gnn_prediction.py  # Placeholder for future GNN implementation
├── utils/
│   └── hex_grid.py    # Utilities for hexagonal grid coordinates
├── visual/
│   └── bokeh_app.py   # Bokeh visualization
└── run_simulation.py  # Entry point script
```

## Key Components

### Network Architecture
- **ε-nodes**: Mobile IoT devices deployed in a hexagonal grid pattern with sonar, hydrophone, and camera sensors
- **σ-nodes**: Fixed relay nodes positioned at strategic locations (currently at 19.5km from shore)
- **θ-contacts**: Marine entities (European seabass, Atlantic horse mackerel, Bottlenose dolphin)

### Protocols
- **Spatiotemporal Contact Volume (SCV)**: Data structure for contact detections
- **Reliable Proximity Relay (RPR)**: Communication protocol between nodes
- **Distributed Observation Buffer (DOB)**: Local and global data storage

### Environmental Dynamics
- **Ocean Currents**: Perlin noise-based vector field
- **Node Drift**: ε-nodes follow local current vectors
- **Marine Motion Models**: Brownian motion for fish, sinusoidal for dolphins

## Important Parameters
- **Geographic Area**: 30km × 16km (from 6km to 22km offshore)
- **Resolution**: 500m spacing (approximately 2,220 ε-nodes)
- **Detection Radius**: 100m (visual indication with color change)
- **Time Step**: 1 second
- **Communication Range**: 10km maximum, 5km optimal

## Setup and Usage
1. Clone the repository
2. Activate the virtual environment:
   ```
   source venv/bin/activate  # On Unix/macOS
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the simulation:
   ```
   python run_simulation.py --show
   ```

## Visualization Features
- Real-time display of node and contact positions
- Visual detection indicator (bright green when contacts are within detection radius)
- Deployment area outline
- Information panel with simulation statistics
- Start/stop controls

## Future Development
### Near-term Goals
- Implement data persistence with HDF5
- Add more sophisticated marine entity behavior
- Improve network topology and communication efficiency

### Long-term Goals
- Implement Graph Neural Network (GNN) for prediction
- Add Kalman filter implementation as a baseline
- Create evaluation metrics for prediction accuracy
- Develop integration with real-world data sources

## Notes for Future Developers
- The core simulation is contained in `simulation_loop.py`, which implements the main tick procedure
- Configuration parameters can be adjusted in `simulation_config.py`
- The visualization is separate from the simulation logic to allow for headless operation
- Detection logic is in `entities.py` in the `EpsilonNode.detect_contact()` method
- Display and actual data collection should remain in sync - entities change color exactly when they're detected

## Research Question
The central research question for the future development is:
Can GNNs better predict multi-entity motion under communication-constrained sensor networks than Kalman-based baselines?
