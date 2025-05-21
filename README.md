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
- ✅ Delaunay-Voronoi mesh network topology for optimal connectivity
- ✅ Real-time visualization with Bokeh
- ✅ Detection visualization (color change for detected entities)
- ✅ Network connections visualization (permanent and intermittent)
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

### Network Topology
The simulation implements a Delaunay-Voronoi hybrid mesh topology for epsilon node connections:

#### Connection Properties:
- **Permanent connections**: Direct links between nodes within 5km (solid blue lines)
- **Intermittent connections**: Links between nodes 5-10km apart (dashed light blue lines)
- **Connection limits**: Nodes maintain 3-5 connections (min: 3, max: 5)
- **Distance constraints**: All connections respect maximum distance limits (10km absolute maximum)

#### Topology Algorithm:
1. **Initial topology (Delaunay Triangulation)**:
   - Creates mathematically optimal mesh network with triangular structure
   - Generates a base topology that naturally includes redundant paths
   - Maintains maximum connection limit per node (5 connections)

2. **Connection Reinforcement**:
   - Ensures all nodes have at least minimum number of connections (3)
   - Identifies and connects network components with bridges
   - All bridges respect maximum distance constraints (10km)
   - Validates full network connectivity after bridge formation

3. **Dynamic Recalculation (Voronoi Properties)**:
   - Updates intermittent connections (5-10km) every 2 hours
   - Uses Voronoi diagram properties to identify natural neighbors
   - Adapts to nodes moving with ocean currents
   - Efficiently handles network evolution without full recalculation

#### Implementation Features:
- Modular design with injectable topology strategies
- Separate `NetworkTopology` class hierarchy for different approaches
- Automatic component detection and bridge formation
- Balance between energy efficiency and network resilience

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
- **Resolution**: 1km spacing (570 ε-nodes)
- **Detection Radius**: 200m (visual indication with color change)
- **Time Step**: 1 second
- **Communication Range**: 10km maximum, 5km optimal
- **Connection Limit**: 3 per node (balancing connectivity and resource usage)

## Research Methodology

We developed a custom marine simulation framework in Python using Bokeh for real-time visualization to evaluate predictive models in the absence of real-world passive sensor networks. The simulated environment spans 30×16 km, with ε-nodes (sensors) deployed in a hexagonal grid and linked via Delaunay triangulation. As the topology evolves, connectivity is updated using Voronoi decomposition. The system models three marine species—two types of fish and cetacea—using Brownian and sinusoidal motion patterns to reflect natural variability and enable trajectory verifiability. We generated 500 independent simulations, from which we extracted 40,000 samples comprising time-series sensor contact data. These were split into training (70%), validation (15%), and test sets (15%), with an additional set of synthetic adversarial cases (e.g., biologically implausible speeds or teleporting entities) used to evaluate model robustness and false positive behavior.

The sensing network was encoded as a heterogeneous, partially observed knowledge graph, with nodes representing sensors and edges representing inferred or observed contact transitions. We framed the predictive task as **link prediction**—learning to infer missing or future species-specific connections between sensors, effectively modeling plausible migratory paths under degraded observability. We implemented a Graph Neural Network using **GraphSAGE** in PyTorch Geometric, leveraging spatiotemporal node features and localized message passing to infer edges. Predictions were evaluated by comparing inferred links to the ground truth movement graph from simulation, using AUC, precision, and recall as primary metrics. Kalman filtering was implemented as a baseline, applied independently per entity to generate continuous position estimates without graph structure awareness.

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
- Visual detection indicator (bright green when detected, grey when not detected)
- Deployment area outline
- Network connections displayed as lines (permanent: solid blue, intermittent: dashed light blue)
- Information panel with simulation statistics
- Start/stop controls
- Simulation speed controls

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
- Network topology is managed in the `_update_epsilon_connections()` method in `simulation_loop.py`

## Research Question
The central research question for the future development is:
Can GNNs better predict multi-entity motion under communication-constrained sensor networks than Kalman-based baselines?