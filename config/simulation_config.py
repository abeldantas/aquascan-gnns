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
ACTIVE_RESOLUTION = "low"  # Choose from RESOLUTIONS keys (back to 1km resolution)

# Communication Parameters
MAX_COMM_RANGE = 10.0  # km (maximum communication range between ε-nodes)
OPTIMAL_COMM_RANGE = 5.0  # km (optimal communication range)
DETECTION_RADIUS = 0.2  # km (200 meters sensor detection radius)

# Temporal Configuration
TIME_STEP = 1.0  # seconds per tick
SIMULATION_DURATION = 60 * 60 * 24 * 30  # 30 days in seconds (extended from 7 days)
SAVE_INTERVAL = 300  # Save simulation state every 5 minutes
MOTION_SUBSTEPS = 5  # Number of substeps for smoother motion calculations
SIMULATION_SPEED = 1.0  # Default: realtime (can be changed via UI)
SIMULATION_START_HOUR = 9  # Simulation starts at 9 AM

# Marine Entity Configuration
MARINE_ENTITIES = {
    "european_seabass": {
        "scientific_name": "Dicentrarchus labrax",
        "motion_model": "brownian",
        "count": 3,
        "typical_volume": 0.01,  # cubic meters
        "speed_range": (0.16, 0.19),  # m/s (approx 14-16 km/day)
        "turn_frequency": 0.002,  # probability of changing direction per second (significantly reduced)
    },
    "atlantic_horse_mackerel": {
        "scientific_name": "Trachurus trachurus",
        "motion_model": "brownian",
        "count": 3,
        "typical_volume": 0.007,  # cubic meters
        "speed_range": (0.19, 0.23),  # m/s (approx 16-20 km/day)
        "turn_frequency": 0.002,  # probability of changing direction per second (significantly reduced)
    },
    "bottlenose_dolphin": {
        "scientific_name": "Tursiops truncatus",
        "motion_model": "sinusoidal",
        "count": 1,  # Less common
        "typical_volume": 3.5,  # cubic meters
        "speed_range": (0.32, 0.39),  # m/s (approx 27-33 km/day)
        "amplitude": 0.2,  # km (increased to create wider paths)
        "period": 3600,  # seconds (increased for much gentler curves)
    }
}

# Environmental Parameters
CURRENT_STRENGTH = 0.03  # Maximum current speed in m/s (increased from 0.01)
CURRENT_VARIABILITY = 0.5  # Variability in current strength (increased from 0.15)
PERLIN_SCALE = 0.03  # Scale factor for Perlin noise (increased from 0.01 - smaller = smoother)
PERLIN_OCTAVES = 3  # Number of octaves for Perlin noise (increased from 2)

# Ocean Current Parameters - Time Cycling
CURRENT_ANGLE_CYCLE_DAYS = 1.0  # Days per full cycle of current angle changes
CURRENT_STRENGTH_CYCLE_DAYS = 0.5  # Days per full cycle of current strength changes
CURRENT_PHASE_OFFSET = 0.25  # Phase offset between angle and strength cycles (0-1)

# Epsilon Node Movement Parameters
EPSILON_NOISE_FACTOR = 0.25  # Individual noise factor (reduced from 0.6)
DISTORTION_FIELD_SCALE = 0.05  # Scale factor for distortion field
SECONDARY_NOISE_FACTOR = 0.3  # Secondary noise layer factor (reduced from 0.8)
SECONDARY_NOISE_FREQUENCY = 0.1  # Frequency of secondary noise pattern
INDEPENDENT_DRIFT_STRENGTH = 0.002  # Strength of independent drift in km/s (reduced from 0.02)
INDEPENDENT_DRIFT_PERSISTENCE = 0.98  # How much previous direction influences new direction (increased slightly)

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
