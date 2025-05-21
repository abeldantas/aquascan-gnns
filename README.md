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
- ✅ Configuration management with OmegaConf
- ✅ Module-based architecture with proper imports
- ✅ Unit testing framework
- ✅ Data persistence with HDF5 snapshots
- ⬜ GNN implementation for prediction
- ⬜ Comparison with Kalman filter baseline

## Project Structure
```
aquascan-gnns/
├── configs/            # YAML configuration files
│   └── base.yml        # Default configuration
├── aquascan/           # Main package
│   ├── config/         # Configuration module
│   │   └── simulation_config.py  # Central configuration file
│   ├── simulation/     # Core simulation components
│   │   ├── ocean_area.py  # Deployment area and node positioning
│   │   ├── entities.py    # ε-nodes, σ-nodes, and θ-contact classes
│   │   ├── sensors.py     # Signal simulation and detection
│   │   ├── communication.py  # Communication protocols (RPR and SCV)
│   │   ├── simulation_loop.py  # Main simulation tick procedure
│   │   └── gnn_prediction.py  # Placeholder for future GNN implementation
│   ├── utils/
│   │   └── hex_grid.py    # Utilities for hexagonal grid coordinates
│   └── run_simulation.py  # Entry point script
├── tests/              # Unit tests
│   └── test_cfg.py     # Tests for configuration functionality
└── requirements.txt    # Project dependencies
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

### Configuration Management
- **OmegaConf**: The project uses OmegaConf for hierarchical configuration management
- **Configuration Files**: YAML format configuration files in the `configs/` directory
- **Command-line Overrides**: Parameters can be overridden via command-line arguments
- **Testing**: Configuration can be mocked for unit testing

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

## Setup and Usage

### Installation
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

#### Interactive Mode (for visual exploration)
Run the simulation with Bokeh visualization:
```bash
python -m aquascan.run_simulation --show
```

#### Single Run (headless mode)
Run a single simulation without visualization:
```bash
python -m aquascan.run_simulation --headless --ticks 600 --seed 42 --out data/raw/42.h5
```

#### Batch Generation

The batch generator enables parallel simulation runs with different random seeds, creating a corpus of HDF5 files for model training and analysis. This is significantly more efficient than running individual simulations and provides a consistent dataset for reproducible research.

Generate multiple simulations in parallel:
```bash
python -m aquascan.batch.generate --cfg configs/base.yml --runs 500 --out data/raw --jobs 8
```

Command-line options for batch generation:
- `--cfg`: Path to the base configuration file (default: configs/base.yml)
- `--runs`: Number of simulations to run (default: 500)
- `--out`: Output directory for HDF5 files (default: data/raw)
- `--jobs`: Number of parallel workers (default: CPU count)
- `--overwrite`: If set, overwrite existing files (default: skip)
- `--validate`: If set, validate files after generation

**Benefits of the batch generator:**
- **Model training & validation**: GNNs and Kalman filters see thousands of distinct trajectories instead of memorizing a single run
- **Benchmark reproducibility**: Training and evaluation can use identical data without rerunning simulations
- **Experiment speed**: Loading pre-written HDF5s is orders of magnitude faster than running physics simulations
- **Parallel efficiency**: Saturates all CPU cores to reduce generation time

**Example workflow:**
1. Generate a large dataset once:
   ```bash
   python -m aquascan.batch.generate --cfg configs/base.yml --runs 500 --out data/raw --jobs 8
   ```

2. Use the dataset for model development and testing:
   ```python
   import h5py
   import glob
   import numpy as np
   
   # Load a batch of simulation files
   simulation_files = glob.glob("data/raw/*.h5")
   
   # Process each file
   for file_path in simulation_files:
       with h5py.File(file_path, "r") as f:
           # Extract nodes and edges for each tick
           nodes = f["nodes"][:]
           edges = f["edges"][:]
           
           # Use the data for model training, validation, etc.
           # ...
   ```

3. Validate generated files:
   ```bash
   python -m aquascan.batch.generate --cfg configs/base.yml --out data/raw --validate
   ```
   
**Implementation details:**
- Generated HDF5 files are named by their seed number (e.g., `0.h5`, `1.h5`, etc.)
- Each file contains complete simulation snapshots with all nodes and edges
- Files are optimized with chunking and compression for efficient access
- Error handling ensures stability during long batch runs

Command-line options:
- `--cfg`: Path to configuration file (default: configs/base.yml)
- `--headless`: Run without visualization
- `--ticks`: Number of simulation steps to run (headless mode only)
- `--seed`: Random seed for reproducibility
- `--port`: Port for Bokeh server (visual mode only)
- `--show`: Open browser automatically (visual mode only)

### Configuration Files
The default configuration file `configs/base.yml` contains:
```yaml
sim:
  ticks: 600
  seed: 42
  resolution_km: 1.0
  area:
    length_km: 30
    width_km: 16
    shore_offset_km: 6
bokeh:
  port: 5006
  show: false
```

### Running Tests
Execute the unit tests:
```bash
python -m pytest tests/
```

Or run a specific test:
```bash
python -m pytest tests/test_cfg.py
```

## Research Methodology

We developed a custom marine simulation framework in Python using Bokeh for real-time visualization to evaluate predictive models in the absence of real-world passive sensor networks. The simulated environment spans 30×16 km, with ε-nodes (sensors) deployed in a hexagonal grid and linked via Delaunay triangulation. As the topology evolves, connectivity is updated using Voronoi decomposition. The system models three marine species—two types of fish and cetacea—using Brownian and sinusoidal motion patterns to reflect natural variability and enable trajectory verifiability. We generated 500 independent simulations, from which we extracted 40,000 samples comprising time-series sensor contact data. These were split into training (70%), validation (15%), and test sets (15%), with an additional set of synthetic adversarial cases (e.g., biologically implausible speeds or teleporting entities) used to evaluate model robustness and false positive behavior.

The sensing network was encoded as a heterogeneous, partially observed knowledge graph, with nodes representing sensors and edges representing inferred or observed contact transitions. We framed the predictive task as **link prediction**—learning to infer missing or future species-specific connections between sensors, effectively modeling plausible migratory paths under degraded observability. We implemented a Graph Neural Network using **GraphSAGE** in PyTorch Geometric, leveraging spatiotemporal node features and localized message passing to infer edges. Predictions were evaluated by comparing inferred links to the ground truth movement graph from simulation, using AUC, precision, and recall as primary metrics. Kalman filtering was implemented as a baseline, applied independently per entity to generate continuous position estimates without graph structure awareness.

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
- Configuration parameters can be adjusted in `simulation_config.py` or via YAML files in `configs/`
- The project uses a modular structure with proper Python package imports
- The simulation supports both headless operation (for data generation/testing) and visual mode
- Detection logic is in `entities.py` in the `EpsilonNode.detect_contact()` method
- OmegaConf is used for configuration management, allowing for hierarchical configurations and overrides
- Unit tests are in the `tests/` directory and use pytest

## Research Question
The central research question for the future development is:
Can GNNs better predict multi-entity motion under communication-constrained sensor networks than Kalman-based baselines?

#### ⛵ Simulation snapshots
Each headless run writes one **self-contained HDF5** file to `data/raw/`
(e.g. `data/raw/42.h5`).  
This file is the "tape" you replay for model training—no need to rerun
the sim unless you change the physics.

| Dataset      | Shape        | dtype   | Description                                                  |
|--------------|--------------|---------|--------------------------------------------------------------|
| **/nodes**   | (N_total,)   | compound| Tick-level node states:<br>  • `t:int32` – tick index<br>  • `gid:int32` – global ID<br>  • `type:uint8` (0 = ε, 1 = θ)<br>  • `x,y:float32` – km coords<br>  • `feat:float32[feat_dim]` – sensor features |
| **/edges**   | (E_total,)   | compound| Tick-level edges:<br>  • `t:int32`, `src:int32`, `dst:int32`<br>  • `rel:uint8` (0 = comm, 1 = detect) |
| **/globals** | scalar       | UTF-8 JSON | Run metadata (seed, ticks, resolution, CLI cfg hash)        |

Both `/nodes` and `/edges` are **chunked on `t` and gzip-compressed**,
so you can stream by tick or memory-map the whole thing.

```bash
# Write a snapshot
python -m aquascan.run_simulation \
       --cfg configs/base.yml \
       --headless \
       --ticks 600 \
       --seed 42 \
       --out data/raw/42.h5

# Read a snapshot
import h5py, json
with h5py.File("data/raw/42.h5") as f:
    nodes = f["nodes"][:]        # NumPy structured array
    edges = f["edges"][:]
    meta  = json.loads(f["globals"][()].decode())
```

> **Tip** — treat snapshots as immutable; if the schema changes,
> write to a new folder rather than mutating existing files.

## Dataset Module

### Graph Building

The `aquascan.dataset` module converts raw simulation data (HDF5) into graph structures for GNN training:

```bash
python -m aquascan.dataset.build_graphs \
       --raw data/raw \
       --out data/processed \
       --context 60 \  # Context window length in ticks
       --horizon 30 \  # Future prediction window length
       --split 0.7 0.15 0.15 \  # train/val/test split
       --adv_fraction 0.05  # Fraction of adversarial examples
```

### Data Structure

- **Raw data**: `data/raw/*.h5` - HDF5 files from simulation containing nodes and edges
- **Processed data**: 
  - `data/processed/train.pt` - Training graphs
  - `data/processed/val.pt` - Validation graphs
  - `data/processed/test.pt` - Test graphs
  - `data/processed/adversarial.pt` - Adversarial examples
  - `data/processed/meta.json` - Dataset metadata

### Graph Format

Each graph is a `HeteroData` object (PyTorch Geometric) with:
- Nodes: `epsilon` (sensors) and `theta` (marine entities)
- Node features: `[x, y, Δx, Δy]` (position and velocity)
- Edge types:
  - `epsilon -> communicates -> epsilon`: Communication links
  - `epsilon -> detects -> theta`: Detection events in context window
  - `epsilon -> will_detect -> theta`: Target edges for prediction (binary labels)

### Developer Notes

- Large binary files (`.h5`, `.pt`) are tracked with Git LFS
- Processing large datasets can be slow (expect minutes to hours depending on size)
- Required dependencies: `torch`, `torch_geometric`, `h5py`, `numpy`
