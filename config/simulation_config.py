"""
Simulation Configuration

This module defines the configurable parameters for the Aquascan simulation.
It serves as a central place to adjust simulation settings including:
- Geographic dimensions
- Sensor network resolution
- Simulation duration and time step
- Marine entity populations
- Environmental parameters

All parameter values can be adjusted here without modifying the core simulation code.
"""

# Geographic Area Configuration
AREA_LENGTH = 30.0  # km (along coastline)
AREA_WIDTH = 16.0   # km (seaward extension)
SHORE_DISTANCE = 6.0  # km (distance from shore to start of area)
# The area starts at SHORE_DISTANCE (6km) and ends at SHORE_DISTANCE + AREA_WIDTH (22km)

# Sensor Network Configuration
RESOLUTIONS = {
    "very_low": 5.0,    # 5km - 24 sensors
    "low": 1.0,         # 1km - 570 sensors
    "medium": 0.5,      # 500m - 2,220 sensors
    "high": 0.1,        # 100m - 55,500 sensors
}
ACTIVE_RESOLUTION = "low"  # Choose from RESOLUTIONS keys

# Communication Parameters
MAX_COMM_RANGE = 10.0  # km (maximum communication range between ε-nodes)
OPTIMAL_COMM_RANGE = 5.0  # km (optimal communication range)
DETECTION_RADIUS = 0.5  # km (500 meters sensor detection radius)

# Temporal Configuration
TIME_STEP = 1.0  # seconds per tick
SIMULATION_DURATION = 60 * 60 * 24 * 7  # 1 week in seconds
SAVE_INTERVAL = 300  # Save simulation state every 5 minutes

# Marine Entity Configuration
MARINE_ENTITIES = {
    "european_seabass": {
        "scientific_name": "Dicentrarchus labrax",
        "motion_model": "brownian",
        "count": 3,
        "typical_volume": 0.01,  # cubic meters
        "speed_range": (0.5, 2.0),  # m/s
        "turn_frequency": 0.1,  # probability of changing direction per second
    },
    "atlantic_horse_mackerel": {
        "scientific_name": "Trachurus trachurus",
        "motion_model": "brownian",
        "count": 3,
        "typical_volume": 0.007,  # cubic meters
        "speed_range": (0.7, 2.5),  # m/s
        "turn_frequency": 0.15,  # probability of changing direction per second
    },
    "bottlenose_dolphin": {
        "scientific_name": "Tursiops truncatus",
        "motion_model": "sinusoidal",
        "count": 1,  # Less common
        "typical_volume": 3.5,  # cubic meters
        "speed_range": (1.5, 6.0),  # m/s
        "amplitude": 0.5,  # km
        "period": 300,  # seconds
    }
}

# Environmental Parameters
CURRENT_STRENGTH = 0.5  # Maximum current speed in m/s
CURRENT_VARIABILITY = 0.2  # Variability in current strength (0-1)
PERLIN_SCALE = 0.1  # Scale factor for Perlin noise (smaller = smoother)
PERLIN_OCTAVES = 3  # Number of octaves for Perlin noise

# Visualization Settings
VIZ_UPDATE_INTERVAL = 100  # milliseconds between visualization updates
PLOT_WIDTH = 1000  # pixels
PLOT_HEIGHT = 600  # pixels
ENTITY_COLORS = {
    "epsilon_node": "#3288bd",  # blue
    "sigma_node": "#fc8d59",    # orange
    "european_seabass": "#99d594",  # green
    "atlantic_horse_mackerel": "#fee08b",  # yellow
    "bottlenose_dolphin": "#d53e4f",  # red
}

def get_resolution():
    """Return the current resolution value in kilometers."""
    return RESOLUTIONS[ACTIVE_RESOLUTION]

def get_node_count():
    """Calculate the number of nodes based on current resolution."""
    resolution = get_resolution()
    
    # Calculate spacing based on hexagonal grid
    vert_spacing = (3**0.5/2) * resolution
    
    # Calculate columns and rows
    columns = int(AREA_LENGTH / resolution)
    rows = int(AREA_WIDTH / vert_spacing)
    
    # Total number of sensors
    return columns * rows

def get_entity_count():
    """Return the total number of marine entities."""
    return sum(entity['count'] for entity in MARINE_ENTITIES.values())

# Print configuration summary when imported
if __name__ == "__main__":
    print(f"Aquascan Simulation Configuration")
    print(f"================================")
    print(f"Area: {AREA_LENGTH}km x {AREA_WIDTH}km at {SHORE_DISTANCE}km from shore")
    print(f"Resolution: {get_resolution()}km")
    print(f"Sensor network: {get_node_count()} ε-nodes")
    print(f"Marine entities: {get_entity_count()} total")
    print(f"Simulation duration: {SIMULATION_DURATION/3600:.1f} hours ({SIMULATION_DURATION/86400:.1f} days)")
    print(f"Time step: {TIME_STEP} seconds")
