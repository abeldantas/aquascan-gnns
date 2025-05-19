---
date: 2025-05-18
tags:
  - loose-note
---

[[2025-05-18 Sunday]]



# Aquascan: A Digital Twin Architecture for Marine Motion Modeling with GNNs
Modeling spatiotemporal behavior of marine life via a distributed ocean twin simulation

I want to write a paper about aquascan, a ocean digital twin that leverages GNNs to predict movement of objects in the sea (namely marine life).

This is a concept of such a system.

The focus of the paper should be the GNN aspect.

Basically we conceive a network, a mesh of low cost IoT devices in the water. Small mobile buoys (ε-node devices) equipped with some sensors: sonar, hydrophones and cameras.

Each ε-node can communicate with others in proximity up until a distance of 10km. But for optimal comms, we want to make sure they are within 5km of each other or less to maintain communication, but to have good resolution on the data we might want them to be as close to each other as 100m.

They collect environmental data, and data on the objects in the water, they then communicate with each other to replicate information that eventually is relayed to a fixed buoys (σ-node). A σ-node might be mounted on an offshore wind turbine or in a speciaized marine vessel that has satelite or 5G connection to relay that information to a datacenter for processing (Ω-node). The data is relayed in realtime (within a threshold of the delays in the transmission between ε-node hops.

Aside from regular data collection of water conditions - for example temperature, salinity, O2 and CO2 dissolved in the water and contaminants. The network collects the raw feed from the sensors to establish patterns for three applications:

- Predict future drift of ε-nodes

- Spaciotemporally map the presence of in-water entities - marine presence over time, these may be objects, manmade or otherwise, vessels or wildlife, we'll refer to these as θ-contact for now on

- Find patterns and predict future θ-contacts

In this paper, we postulate that a heterogenous graph neural network will outperform the conventional modeling techniques used in marine monitoring systems. Within the scope of this paper we will use the Kalman filtering as the baseline since it is the de facto standard in montion prediction and widely usied in marine and UAV literature.

To address this gap—and since our proposed system is conceptual—we developed an interactive simulation environment that models a realistic deployment of ε- and σ-nodes. Within this framework, we also simulated a basic population of θ-contacts, representing in-water entities such as fish and cetaceans. While the modeling of these contacts is still in early stages—given that very little is known at the granularity, resolution, and temporal scale we aim to capture—it provides a foundation for evaluating learning models under the kinds of complex, multi-modal conditions expected in real deployments.

This simulation produces a synthetic dataset designed to reflect the communication constraints, sensor modalities, and environmental dynamics of a distributed marine sensing system. Along with it, we release a modeling framework to support future work in this area, enabling researchers to replicate our experiments and extend the study of learning-based approaches for spatiotemporal reasoning at sea.

RQ: Can GNNs better predict multi-entity motion under communication-constrained sensor networks than Kalman-based baselines?

---

Graph Structure:

```
Nodes:
  - ε-node (IoT device, fixed features + temporal data)
  - σ-node (relay, optional modeling)
  - θ-contact (mobile entity, inferred position over time)

Edges:
  - ε–ε communication (based on range)
  - ε–θ contact (if detected in sensor range)
  - ε–σ relay (fixed routes)
```

---

Steps:
1. Build the Simulation
2. Design the Dataset Format
3. Implement the Baseline
4. Prototype the GNN
5. Setup Evaluation Metrics

---

## Build the Simulation

### Geography

We define the deployment area as a rectangular offshore strip measuring **30 km along the coastline** and extending **16 km seaward**, starting **6 km from shore** and ending **22 km offshore**. This area intentionally excludes the nearshore artisanal fishing zone (typically within 6 km) and focuses on the operational range of **small to medium-scale fishing vessels** (roughly 6–12 km offshore), as well as known **marine wildlife corridors** that span from the edge of coastal activity out to offshore infrastructure zones.

The selected area thus captures a gradient from nearshore ecological activity through mid-depth fishing grounds and into offshore environments relevant for both biodiversity monitoring and technological deployments (e.g., wind farms or technological free zones).

We should note that the same area could be applied to a ring-type topology modelling the system around an island, however, for modelling purposes a rectangular layout is easier to define uniform overlap zones since we can apply a hex pattern without having to resort to wrapping or curvature, making the initial configuration of the grid easier to visualize. Furthermore, a strip layout allows for extention in one dimension without having to redisign the topology - making it more scalable by default.

In summary:

```
- **Region Type**: Rectangular offshore strip    
- **Dimensions**: 30 km (coastline-aligned) × 16 km (seaward extension)
- **Grid Topology**: Hexagonal layout, optimized for coverage and scalability
```

### Sensor Resolution

The volume of ε-nodes varies with the desired resolution of the Aquascan:

| Resolution          | Horizontal Spacing | Vertical Spacing | Columns | Rows | Total Sensors |
| ------------------- | ------------------ | ---------------- | ------- | ---- | ------------- |
| 5 km                | 5.0 km             | 4.33 km          | 6       | 4    | 24            |
| 1 km                | 1.0 km             | 0.87 km          | 30      | 19   | 570           |
| 500 meters (0.5 km) | 0.5 km             | 0.43 km          | 60      | 37   | 2220          |
| 100 meters (0.1 km) | 0.1 km             | 0.09 km          | 300     | 185  | 55,500        |
The resolution table assumes a **hexagonal grid**, where each sensor has six neighbors. Vertical spacing follows:
vertical spacing=32×horizontal spacing\text{vertical spacing} = \frac{\sqrt{3}}{2} \times \text{horizontal spacing}vertical spacing=23​​×horizontal spacing
This layout maximizes coverage while respecting distance constraints (and needs 13.4% less sensors than a rectangular grid).

Let's initially consider a 1km resolution of 2,220 ε-nodes and 3 σ-nodes.


### Core Protocols

#### A. Spatiotemporal Contact Volume (SCV)

SCV is a formal data structure that captures the estimated volume and absolute position of a detected entity (theta-contact) in space and time. It represents the output of multi-modal sensing systems (e.g., sonar, cameras, hydrophones), abstracted into a unified event record.

Unlike traditional sensing APIs, SCV intentionally **abstracts away uncertainty**. Rather than expressing confidence scores or detection probabilities, the protocol assumes a detection has occurred and encodes the spatiotemporal bounds and entity type directly. This decision prioritizes deterministic post-processing over in-sensor ambiguity handling, shifting uncertainty management to downstream systems such as probabilistic filters or GNNs.

Each SCV includes an **absolute 2D position** for the detection. While epsilon nodes (mobile sensors) are conceived as having GPS hardware, they may not always know their exact coordinates in real time (e.g., due to drift or loss of fix). However, by leveraging the Reliable Proximity Relay (RPR) protocol (defined below), we assume that position can be estimated or corrected in near-real-time through neighbor consensus or network triangulation.

Example structure:

```json
{
  "epsilon_id": "e-0231",
  "timestamp": 123456789,
  "theta_id": "θ-089",
  "position": { "x": 14432.7, "y": 8312.4 },
  "estimated_volume": 17.3,
  "entity_type": "Dicentrarchus labrax"
}
```

#### B. Reliable Proximity Relay (RPR)

RPR defines the communication behavior of epsilon nodes as they transmit data across a short-range, hop-based network. RPR ensures that all relevant contact data (e.g., SCVs) is propagated efficiently to relay nodes (sigma-nodes) or neighboring sensors, respecting temporal and topological constraints.

Key properties:
- Operates in a **bounded range** (ideal ≤5 km, max 10 km)
- Assumes **partial synchrony** — messages eventually reach a relay node within a bounded delay (Δt)
- Designed for **non-blocking**, lossy environments
- Supports **local buffering and retransmission** in case of network instability

RPR provides the basis for real-time geolocation correction, by allowing epsilon nodes to infer or correct their absolute position based on neighboring state.

#### C. Distributed Observation Buffer (DOB)

DOB is the logging layer responsible for storing all SCVs and node metadata (position, drift vector, etc.) in a time-ordered structure. Each epsilon node maintains its own local buffer, which is periodically offloaded via RPR to a sigma-node or directly to the central Ω-node.

---

### Node Properties

#### A. Epsilon Node (ε-node)
- `id`: Unique node identifier
- `position`: Known or network-inferred (2D x, y)
- `scv_emitter`: Emits SCVs upon θ-contact
- `rpr_transceiver`: Handles communication via RPR
- `drift_model`: Simulated ocean current + Perlin noise
- `detection_radius`: 100 meters
- `neighborhood`: Dynamic, determined by RPR range
- `dob`: Local instance of the Distributed Observation Buffer

#### B. Sigma Node (σ-node)
    - `position`: Fixed (e.g., offshore wind turbine)
    - `uplink`: Persistent to Ω-node (e.g., via 5G/satellite)
    - `relay_radius`: Large (≈10 km)
    - `buffer`: Aggregates SCVs and node telemetry

---

### Theta-Contact Entities (θ)

- **Entities**:
    - _European seabass_ (_Dicentrarchus labrax_)
    - _Atlantic horse mackerel_ (_Trachurus trachurus_)
    - _Bottlenose dolphin_ (_Tursiops truncatus_)
- **Motion Models**:
    - Fish schools → Brownian motion
    - Dolphins → Sinusoidal trajectory
- **Trigger Condition**:
    - Any θ within 100m of ε-node generates a SCV

---

### Environmental Dynamics

- **Ocean Currents**: Perlin noise-based vector field
- **Node Drift**: ε-nodes follow local current field
- **Clock Model**: Partial synchrony assumed
- **Time Model**: Discrete, globally-stepped ticks

---

### 6. Simulation Loop

```simulation-loop: procedure TICK(t)
2:     if ocean_current is dynamic then
3:         update ocean_current_field
4:
5:     for each ε in epsilon_nodes do
6:         v ← ocean_current_vector_at(ε.position)
7:         ε.position ← ε.position + v
8:
9:     for each θ in theta_contacts do
10:        update θ.position according to motion_model
11:
12:    for each ε in epsilon_nodes do
13:        for each θ in theta_contacts do
14:            if distance(ε.position, θ.position) ≤ detection_radius then
15:                scv ← generate_scv(ε, θ, t)
16:                send scv to σ via RPR
17:
18:    for each ε in epsilon_nodes do
19:        append scv and telemetry to DOB
20:
21:    if visualization_enabled then
22:        update_visualization

```

---

### Tools used for simulation

Simulation Engine: NumPy, NetworkX, custom logic

Data Model/Graph:`NetworkX`, optional PyTorch Geometric

Real-time visual output: Bokeh

Backend loop: bokeh + PeriodicCallback

Persistence: HDF5 (h5py)

Project Structure:
```
aquascan/
├── simulation/
│   ├── ocean_area.py         # defines hex grid, ε-/σ-node positioning
│   ├── entities.py           # θ-contact behavior
│   ├── sensors.py            # signal simulation, detection
│   ├── communication.py      # ε-ε link logic
│   └── simulation_loop.py    # master tick loop
├── visual/
│   └── bokeh_app.py          # Bokeh layout, callbacks, real-time display
├── data/
│   └── log_writer.py         # to persist timestep data
├── utils/
│   └── hex_grid.py           # coordinate math
├── README.md
└── requirements.txt
```

