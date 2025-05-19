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
from utils.hex_grid import get_deployment_positions
from config.simulation_config import (
    AREA_LENGTH, AREA_WIDTH, SHORE_DISTANCE,
    get_resolution, MAX_COMM_RANGE, OPTIMAL_COMM_RANGE,
    PERLIN_SCALE, PERLIN_OCTAVES, CURRENT_STRENGTH, CURRENT_VARIABILITY
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
        positions = get_deployment_positions(self.length, self.width, self.resolution)
        print(f"Deployed {len(positions)} ε-nodes with resolution {self.resolution}km")
        return positions
    
    def _deploy_sigma_nodes(self):
        """Position σ-nodes at strategic locations for relay purposes."""
        # Place 3 sigma nodes along the shoreline side (x=0) at 1/4, 1/2, and 3/4 of the length
        positions = np.array([
            [0, self.width * 0.25],
            [0, self.width * 0.5],
            [0, self.width * 0.75]
        ])
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
        nt = time / 86400 * PERLIN_SCALE  # Normalize time (days)
        
        # Calculate Perlin noise value for angle
        angle_noise = noise.pnoise3(nx, ny, nt, octaves=PERLIN_OCTAVES)
        angle = angle_noise * 2 * np.pi  # Convert to angle in radians
        
        # Calculate Perlin noise value for strength
        strength_noise = noise.pnoise3(nx + 10.0, ny + 10.0, nt, octaves=PERLIN_OCTAVES)
        strength = CURRENT_STRENGTH * (1.0 + strength_noise * CURRENT_VARIABILITY)
        
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
        return 0 <= x <= self.length and 0 <= y <= self.width
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions in km."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
