"""
Ocean Area Module

This module defines the simulation area and handles node positioning.
Responsibilities:
- Define the geographic boundaries of the simulation area
- Deploy ε-nodes in a hexagonal grid pattern
- Position σ-nodes (relay nodes) at strategic locations
- Initialize the ocean current vector field
- Calculate distances and connections between nodes
"""

import numpy as np
import noise
from aquascan.utils.hex_grid import get_deployment_positions
from aquascan.config.simulation_config import (
    AREA_LENGTH, AREA_WIDTH, SHORE_DISTANCE,
    get_resolution, MAX_COMM_RANGE, OPTIMAL_COMM_RANGE,
    PERLIN_SCALE, PERLIN_OCTAVES, CURRENT_STRENGTH, CURRENT_VARIABILITY,
    CURRENT_ANGLE_CYCLE_DAYS, CURRENT_STRENGTH_CYCLE_DAYS, CURRENT_PHASE_OFFSET
)


class OceanArea:
    """Represents the geographic area where the simulation takes place."""
    
    def __init__(self):
        """Initialize the ocean area with sensor network and environmental parameters."""
        self.length = AREA_LENGTH
        self.width = AREA_WIDTH
        self.shore_distance = SHORE_DISTANCE
        self.resolution = get_resolution()
        
        # Generate node positions
        self.epsilon_positions = self._deploy_epsilon_nodes()
        self.sigma_positions = self._deploy_sigma_nodes()
        
        # Initialize Perlin noise for ocean currents
        self.perlin_seed = np.random.randint(0, 1000)
        
        # Calculate connections between nodes
        self.connections = self._calculate_connections()
    
    def _deploy_epsilon_nodes(self):
        """Deploy ε-nodes in a hexagonal grid pattern."""
        # Create a grid that starts at SHORE_DISTANCE from shore
        positions = get_deployment_positions(self.length, self.width, self.resolution)
        
        # Offset the positions to start at SHORE_DISTANCE from shore
        # This shifts all y-coordinates by SHORE_DISTANCE
        for i in range(len(positions)):
            positions[i][1] += self.shore_distance
            
        print(f"Deployed {len(positions)} ε-nodes with resolution {self.resolution}km")
        print(f"  - Area spans from {self.shore_distance}km to {self.shore_distance + self.width}km from shore")
        print(f"  - Area extends {self.length}km along the coastline")
        
        return positions
    
    def _deploy_sigma_nodes(self):
        """Position σ-nodes at strategic locations for relay purposes."""
        # Place sigma nodes at the specified positions:
        # 1. At 20km from shore and 15km along the coastline
        # 2. At 20km from shore and at the left edge (x=0)
        # 3. At 20km from shore and at the right edge (x=AREA_LENGTH)
        positions = np.array([
            [15, 20],                 # Middle node at x=15, y=20
            [0, 20],                  # Left edge node at x=0, y=20
            [self.length, 20]         # Right edge node at x=AREA_LENGTH, y=20
        ])
        
        print(f"Deployed σ-nodes at positions:")
        for pos in positions:
            print(f"  - ({pos[0]:.1f}, {pos[1]:.1f}) km")
            
        return positions
    
    def _calculate_connections(self):
        """Calculate connections between ε-nodes based on communication range."""
        # Dictionary to store connections
        connections = {}
        
        # For each ε-node, find all other ε-nodes within OPTIMAL_COMM_RANGE
        for i, pos_i in enumerate(self.epsilon_positions):
            connections[i] = []
            for j, pos_j in enumerate(self.epsilon_positions):
                if i != j:
                    distance = np.linalg.norm(pos_i - pos_j)
                    # Only connect if within optimal communication range
                    if distance <= OPTIMAL_COMM_RANGE:
                        connections[i].append(j)
        
        return connections
    
    def calculate_ocean_current(self, position, time):
        """
        Calculate ocean current vector at a given position and time.
        
        Args:
            position: (x, y) position in km
            time: Simulation time in seconds
            
        Returns:
            tuple: (dx, dy) current velocity vector in m/s
        """
        # Get normalized coordinates for Perlin noise
        nx = position[0] / self.length * PERLIN_SCALE
        ny = position[1] / self.width * PERLIN_SCALE
        
        # Use different time scales and patterns for different components
        # This creates more varied and realistic patterns over time
        seconds_per_day = 86400
        
        # Time component for angle - uses specified cycle in days
        nt_angle = time / (seconds_per_day * CURRENT_ANGLE_CYCLE_DAYS) 
        
        # Time component for strength - different cycle than angle
        nt_strength = time / (seconds_per_day * CURRENT_STRENGTH_CYCLE_DAYS) + CURRENT_PHASE_OFFSET
        
        # More complex angle variation 
        # Combine two cycles with different frequencies
        angle_base = np.sin(2 * np.pi * nt_angle) * 2 * np.pi
        angle_mod = np.sin(2 * np.pi * nt_angle * 2.7) * np.pi / 2  # Secondary frequency
        
        # Calculate Perlin noise value for position-based variation
        pos_angle_mod = noise.pnoise2(nx, ny, octaves=PERLIN_OCTAVES) * np.pi / 2
        
        # Combine time cycles with position variation
        angle = angle_base + angle_mod + pos_angle_mod
        
        # Calculate more complex strength variation
        # Combine absolute sine wave with Perlin noise
        strength_time_factor = abs(np.sin(2 * np.pi * nt_strength)) * 0.8 + 0.2  # Range 0.2-1.0
        
        # Position-based strength variation 
        strength_pos_mod = noise.pnoise2(nx + 10.0, ny + 10.0, octaves=PERLIN_OCTAVES)
        
        # Combine time and position variations
        strength = CURRENT_STRENGTH * strength_time_factor * (1.0 + strength_pos_mod * CURRENT_VARIABILITY)
        
        # Calculate current vector
        dx = strength * np.cos(angle)
        dy = strength * np.sin(angle)
        
        return (dx, dy)
    
    def get_node_neighbors(self, node_index):
        """Get indices of neighboring nodes for a given ε-node."""
        return self.connections.get(node_index, [])
    
    def is_within_bounds(self, position):
        """Check if a position is within the simulation area."""
        x, y = position
        # Check if within x bounds (0 to length)
        x_ok = 0 <= x <= self.length
        # Check if within y bounds (shore_distance to shore_distance + width)
        y_ok = self.shore_distance <= y <= (self.shore_distance + self.width)
        return x_ok and y_ok
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions in km."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
