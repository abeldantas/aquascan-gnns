"""
Entities Module

This module defines the various entities in the Aquascan simulation.
Responsibilities:
- Define ε-nodes (sensor nodes)
- Define σ-nodes (relay nodes)
- Define θ-contacts (marine entities)
- Implement motion models for entities
- Handle entity state updates

Each entity type has specific properties and behaviors that reflect its role
in the marine monitoring system.
"""

import numpy as np
import noise
from config.simulation_config import (
    DETECTION_RADIUS, MARINE_ENTITIES,
    MAX_COMM_RANGE, TIME_STEP, MOTION_SUBSTEPS,
    SIMULATION_SPEED, EPSILON_NOISE_FACTOR, DISTORTION_FIELD_SCALE,
    SECONDARY_NOISE_FACTOR, SECONDARY_NOISE_FREQUENCY,
    INDEPENDENT_DRIFT_STRENGTH, INDEPENDENT_DRIFT_PERSISTENCE
)


class BaseEntity:
    """Base class for all entities in the simulation."""
    
    def __init__(self, entity_id, position):
        """
        Initialize a base entity.
        
        Args:
            entity_id (str): Unique identifier for the entity
            position (tuple): Initial (x, y) position in km
        """
        self.id = entity_id
        self.position = np.array(position, dtype=float)
        self.creation_time = 0
    
    def update(self, current_time, ocean_area):
        """
        Update the entity state based on simulation time and environment.
        
        Args:
            current_time (float): Current simulation time in seconds
            ocean_area (OceanArea): Reference to the ocean area object
        """
        pass
    
    def get_position(self):
        """Get the current position of the entity."""
        return self.position


class EpsilonNode(BaseEntity):
    """
    ε-node: Mobile IoT sensor device.
    
    These nodes drift with ocean currents and detect marine entities.
    They communicate with each other and relay data to σ-nodes.
    """
    
    def __init__(self, entity_id, position):
        """Initialize an ε-node with sensor and communication capabilities."""
        super().__init__(f"e-{entity_id}", position)
        self.type = "epsilon_node"
        self.detection_radius = DETECTION_RADIUS
        self.detected_contacts = {}  # Dictionary to store detected contacts
        self.dob = []  # Distributed Observation Buffer (local data storage)
        self.last_update_time = 0
        
        # Initialize independent movement direction
        self.independent_direction = np.random.uniform(0, 2 * np.pi)
        self.independent_speed = np.random.uniform(0.8, 1.2) * INDEPENDENT_DRIFT_STRENGTH
    
    def update(self, current_time, ocean_area):
        """Update ε-node position based on ocean currents with added randomness."""
        # Calculate time delta since last update
        dt = current_time - self.last_update_time
        if dt <= 0:
            return
        
        # === LAYER 1: OCEAN CURRENT ===
        # Get current at this position
        current_vector = ocean_area.calculate_ocean_current(self.position, current_time)
        
        # Scale to km and apply time step - convert from m/s to km/s
        # Note: dt already includes SIMULATION_SPEED effects from the tick() method
        drift = np.array(current_vector) * dt * 0.001  # m/s to km/s
        
        # === LAYER 2: DISTORTION FIELD ===
        # Generate a distortion field based on position
        # This creates spatially varied currents - nodes in different regions behave differently
        nx = self.position[0] / ocean_area.length * DISTORTION_FIELD_SCALE * 10
        ny = self.position[1] / ocean_area.width * DISTORTION_FIELD_SCALE * 10
        
        # Use noise for distortion
        distortion_factor = 0.5 + noise.pnoise2(nx, ny, octaves=2)  # Range approximately 0-1
        
        # Apply the distortion factor to the drift vector
        drift = drift * distortion_factor
        
        # === LAYER 3: SECONDARY NOISE ===
        # Generate a secondary movement pattern that's different for each epsilon
        nx2 = self.position[0] / ocean_area.length * SECONDARY_NOISE_FREQUENCY * 10
        ny2 = self.position[1] / ocean_area.width * SECONDARY_NOISE_FREQUENCY * 10
        # Use node ID as a third dimension to ensure different patterns per node
        nz2 = float(self.id.replace('e-', '')) * 0.1  
        
        # Generate noise-based secondary vector (perpendicular to current)
        sec_noise_val = noise.pnoise3(nx2, ny2, nz2, octaves=2) * 2 - 1  # Range -1 to 1
        
        # Create perpendicular vector to current
        perp_x = -drift[1]
        perp_y = drift[0]
        
        # Normalize and scale
        if (perp_x != 0 or perp_y != 0):
            perp_mag = np.sqrt(perp_x**2 + perp_y**2)
            perp_x /= perp_mag
            perp_y /= perp_mag
            
            # Scale by drift magnitude and noise factor
            # This time make it a much stronger factor
            secondary_drift = np.array([
                perp_x * np.linalg.norm(drift) * SECONDARY_NOISE_FACTOR * sec_noise_val,
                perp_y * np.linalg.norm(drift) * SECONDARY_NOISE_FACTOR * sec_noise_val
            ])
            
            # Add to drift vector
            drift = drift + secondary_drift
        
        # === LAYER 4: PERSISTENT INDEPENDENT MOVEMENT ===
        # Gradually change the independent direction with some randomness
        direction_change = np.random.normal(0, 0.01)  # Reduced from 0.1
        self.independent_direction = (
            self.independent_direction * INDEPENDENT_DRIFT_PERSISTENCE + 
            direction_change * (1 - INDEPENDENT_DRIFT_PERSISTENCE)
        ) % (2 * np.pi)
        
        # Calculate independent movement vector
        independent_dx = np.cos(self.independent_direction) * self.independent_speed * dt
        independent_dy = np.sin(self.independent_direction) * self.independent_speed * dt
        independent_drift = np.array([independent_dx, independent_dy])
        
        # === LAYER 5: RANDOM NOISE ===
        # Add individual noise to movement (with magnitude based on EPSILON_NOISE_FACTOR)
        noise_magnitude = np.linalg.norm(drift) * EPSILON_NOISE_FACTOR
        individual_noise = np.random.normal(0, noise_magnitude, 2)  # 2D random vector
        
        # === COMBINE ALL MOVEMENT COMPONENTS ===
        # Update position with all layers of movement
        self.position += drift + independent_drift + individual_noise
        
        # No boundary checking - allow nodes to move freely
        
        self.last_update_time = current_time
    
    def detect_contact(self, contact, current_time):
        """
        Detect a marine entity if it's within detection radius.
        
        Args:
            contact (ThetaContact): Marine entity to detect
            current_time (float): Current simulation time
            
        Returns:
            bool: True if the contact was detected, False otherwise
        """
        distance = np.linalg.norm(self.position - contact.position)
        
        if distance <= self.detection_radius:
            # Generate SCV (Spatiotemporal Contact Volume)
            scv = {
                "epsilon_id": self.id,
                "timestamp": current_time,
                "theta_id": contact.id,
                "position": contact.position.copy(),
                "estimated_volume": contact.volume,
                "entity_type": contact.species_name,
                "distance": distance  # Store the distance for analysis
            }
            
            # Store in Distributed Observation Buffer
            self.dob.append(scv)
            
            # Track this contact
            self.detected_contacts[contact.id] = current_time
            
            return True
        
        return False


class SigmaNode(BaseEntity):
    """
    σ-node: Relay station with persistent connectivity.
    
    These nodes have fixed positions and relay data from ε-nodes to the central system.
    """
    
    def __init__(self, entity_id, position):
        """Initialize a σ-node with relay capabilities."""
        super().__init__(f"s-{entity_id}", position)
        self.type = "sigma_node"
        self.relay_radius = MAX_COMM_RANGE
        self.buffer = []  # Buffer for collected data
    
    def collect_data(self, epsilon_node):
        """
        Collect data from an ε-node if it's within relay radius.
        
        Args:
            epsilon_node (EpsilonNode): Node to collect data from
            
        Returns:
            bool: True if data was collected, False otherwise
        """
        distance = np.linalg.norm(self.position - epsilon_node.position)
        
        if distance <= self.relay_radius:
            # Copy data from epsilon node's DOB to sigma node's buffer
            self.buffer.extend(epsilon_node.dob)
            
            # Clear epsilon node's buffer after collection
            epsilon_node.dob = []
            
            return True
        
        return False


class ThetaContact(BaseEntity):
    """
    θ-contact: Marine entity (fish, dolphins, etc.).
    
    These are the entities being detected and tracked by the sensor network.
    """
    
    def __init__(self, entity_id, species_type, position):
        """
        Initialize a marine entity.
        
        Args:
            entity_id (str): Unique identifier
            species_type (str): Type of marine entity (key in MARINE_ENTITIES config)
            position (tuple): Initial (x, y) position
        """
        super().__init__(f"θ-{entity_id}", position)
        self.type = species_type
        self.species_name = MARINE_ENTITIES[species_type]["scientific_name"]
        self.motion_model = MARINE_ENTITIES[species_type]["motion_model"]
        self.volume = MARINE_ENTITIES[species_type]["typical_volume"]
        self.last_update_time = 0  # Initialize last_update_time
        
        # Motion parameters
        species_config = MARINE_ENTITIES[species_type]
        
        if self.motion_model == "brownian":
            # Parameters for Brownian motion
            min_speed, max_speed = species_config["speed_range"]
            self.speed = np.random.uniform(min_speed, max_speed) * 0.001  # Convert to km/s
            self.direction = np.random.uniform(0, 2 * np.pi)
            self.turn_frequency = species_config["turn_frequency"]
            
        elif self.motion_model == "sinusoidal":
            # Parameters for sinusoidal motion
            min_speed, max_speed = species_config["speed_range"]
            self.speed = np.random.uniform(min_speed, max_speed) * 0.001  # Convert to km/s
            self.amplitude = species_config["amplitude"]
            self.period = species_config["period"]
            self.phase = np.random.uniform(0, 2 * np.pi)
            self.base_direction = np.random.uniform(0, 2 * np.pi)
    
    def update(self, current_time, ocean_area):
        """Update marine entity position based on its motion model."""
        # Calculate time delta
        dt = current_time - self.last_update_time
        if dt <= 0:
            # Skip update if no time has passed
            return
            
        if self.motion_model == "brownian":
            self._update_brownian(current_time, dt, ocean_area)
        elif self.motion_model == "sinusoidal":
            self._update_sinusoidal(current_time, dt, ocean_area)
            
        # Update last update time
        self.last_update_time = current_time
    
    def _update_brownian(self, current_time, dt, ocean_area):
        """Update position using Brownian motion model with substeps for smoother movement."""
        # Use substeps for smoother motion
        sub_time_step = dt / MOTION_SUBSTEPS
        
        for _ in range(MOTION_SUBSTEPS):
            # Randomly change direction with some probability
            if np.random.random() < self.turn_frequency * sub_time_step:
                # Change direction by a smaller random angle for more gradual turns
                self.direction += np.random.normal(0, np.pi/12)  # Reduced from pi/4 to pi/12
                self.direction %= 2 * np.pi
            
            # Calculate movement vector for this substep
            dx = self.speed * np.cos(self.direction) * sub_time_step
            dy = self.speed * np.sin(self.direction) * sub_time_step
            
            # Apply movement
            new_position = self.position + np.array([dx, dy])
            
            # Modified boundary checking to allow wider migration paths
            # We'll only change direction if they get very far outside the deployment area
            # +/- 15km outside the normal bounds
            x, y = new_position
            extended_x_min, extended_x_max = -15, ocean_area.length + 15
            extended_y_min, extended_y_max = ocean_area.shore_distance - 15, ocean_area.shore_distance + ocean_area.width + 15
            
            if x < extended_x_min or x > extended_x_max or y < extended_y_min or y > extended_y_max:
                # Only if we're way outside bounds, turn toward center of deployment area
                center_x = ocean_area.length / 2
                center_y = ocean_area.shore_distance + ocean_area.width / 2
                
                # Calculate angle to center
                angle_to_center = np.arctan2(center_y - y, center_x - x)
                
                # Gradually adjust direction (blend current with target)
                angle_diff = ((angle_to_center - self.direction + np.pi) % (2 * np.pi)) - np.pi
                self.direction += angle_diff * 0.1  # Gradual adjustment
                self.direction %= 2 * np.pi
            
            # Always apply movement regardless of bounds (allow them to leave deployment area)
            self.position = new_position
    
    def _update_sinusoidal(self, current_time, dt, ocean_area):
        """Update position using sinusoidal motion model with substeps for smoother movement."""
        # Use substeps for smoother motion
        sub_time_step = dt / MOTION_SUBSTEPS
        
        # Starting time for this update (for consistent phase calculation)
        step_time = current_time - dt
        
        for i in range(MOTION_SUBSTEPS):
            # Calculate current time for this substep
            substep_time = step_time + (i + 1) * sub_time_step
            
            # Calculate phase for sinusoidal motion
            phase_time = substep_time % self.period
            phase_factor = (phase_time / self.period) * 2 * np.pi + self.phase
            sine_factor = np.sin(phase_factor)
            
            # Calculate the direction offset from the base direction
            direction_offset = sine_factor * np.pi/6  # Max 30 degree deviation
            current_direction = self.base_direction + direction_offset
            
            # Calculate movement vector with the intended speed
            dx = self.speed * np.cos(current_direction) * sub_time_step
            dy = self.speed * np.sin(current_direction) * sub_time_step
            
            # Apply movement
            new_position = self.position + np.array([dx, dy])
            
            # Modified boundary checking to allow wider migration paths
            x, y = new_position
            extended_x_min, extended_x_max = -15, ocean_area.length + 15
            extended_y_min, extended_y_max = ocean_area.shore_distance - 15, ocean_area.shore_distance + ocean_area.width + 15
            
            if x < extended_x_min or x > extended_x_max or y < extended_y_min or y > extended_y_max:
                # If far outside bounds, gradually turn toward center of deployment area
                center_x = ocean_area.length / 2
                center_y = ocean_area.shore_distance + ocean_area.width / 2
                
                # Calculate angle to center
                angle_to_center = np.arctan2(center_y - y, center_x - x)
                
                # Gradually adjust direction
                angle_diff = ((angle_to_center - self.base_direction + np.pi) % (2 * np.pi)) - np.pi
                self.base_direction += angle_diff * 0.1  # Gradual adjustment
                self.base_direction %= 2 * np.pi
                
                # Reset phase for smoother transition
                self.phase = np.random.uniform(0, 2 * np.pi)
            
            # Update position
            self.position = new_position
