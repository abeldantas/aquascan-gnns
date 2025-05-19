"""
Simulation Loop Module

This module implements the main simulation loop for the Aquascan system.
Responsibilities:
- Initialize all simulation components
- Implement the tick procedure
- Update entity positions and states
- Handle detection of marine entities
- Process communication between nodes
- Coordinate with visualization
"""

import numpy as np
import time

from config.simulation_config import (
    TIME_STEP, SIMULATION_DURATION, MARINE_ENTITIES,
    ACTIVE_RESOLUTION, AREA_LENGTH, AREA_WIDTH, SHORE_DISTANCE,
    SIMULATION_SPEED
)
from simulation.ocean_area import OceanArea
from simulation.entities import EpsilonNode, SigmaNode, ThetaContact
from simulation.communication import ReliableProximityRelay, DistributedObservationBuffer


class AquascanSimulation:
    """
    Main simulation class that coordinates all components and runs the simulation.
    """
    
    def __init__(self):
        """Initialize the simulation with all required components."""
        # Initialize the ocean area
        self.ocean_area = OceanArea()
        
        # Initialize nodes
        self.epsilon_nodes = self._create_epsilon_nodes()
        self.sigma_nodes = self._create_sigma_nodes()
        
        # Initialize marine entities
        self.theta_contacts = self._create_theta_contacts()
        
        # Initialize communication protocols
        self.rpr = ReliableProximityRelay(self.ocean_area)
        self.dob = DistributedObservationBuffer()
        
        # Simulation state
        self.current_time = 0
        self.is_running = False
        self.visualization_callback = None
        
        # Statistics
        self.stats = {
            "detections": 0,
            "messages_sent": 0,
            "messages_delivered": 0
        }
    
    def _create_epsilon_nodes(self):
        """Create ε-nodes based on hexagonal grid positions."""
        nodes = []
        for i, position in enumerate(self.ocean_area.epsilon_positions):
            nodes.append(EpsilonNode(str(i).zfill(4), position))
        return nodes
    
    def _create_sigma_nodes(self):
        """Create σ-nodes at strategic positions."""
        nodes = []
        for i, position in enumerate(self.ocean_area.sigma_positions):
            nodes.append(SigmaNode(str(i).zfill(4), position))
        return nodes
    
    def _create_theta_contacts(self):
        """Create marine entities based on configuration."""
        contacts = []
        contact_id = 0
        
        for species_type, config in MARINE_ENTITIES.items():
            for i in range(config["count"]):
                # Random position within the ocean area (accounting for shore_distance)
                position = [
                    np.random.uniform(0, AREA_LENGTH),
                    np.random.uniform(SHORE_DISTANCE, SHORE_DISTANCE + AREA_WIDTH)
                ]
                
                # Create the contact
                contacts.append(
                    ThetaContact(str(contact_id).zfill(3), species_type, position)
                )
                contact_id += 1
        
        return contacts
    
    def register_visualization(self, callback):
        """
        Register a callback function for visualization updates.
        
        Args:
            callback: Function to call with simulation state for visualization
        """
        self.visualization_callback = callback
    
    def initialize(self):
        """Set up the simulation before starting."""
        # Reset time and statistics
        self.current_time = 0
        self.stats = {
            "detections": 0,
            "messages_sent": 0,
            "messages_delivered": 0
        }
        
        # Initialize nodes with starting time
        for node in self.epsilon_nodes + self.sigma_nodes:
            node.creation_time = self.current_time
            if hasattr(node, 'last_update_time'):
                node.last_update_time = self.current_time
        
        # Initialize θ-contacts
        for contact in self.theta_contacts:
            contact.creation_time = self.current_time
    
    def start(self):
        """Start the simulation."""
        self.is_running = True
        print(f"Simulation started with {len(self.epsilon_nodes)} ε-nodes, "
              f"{len(self.sigma_nodes)} σ-nodes, and {len(self.theta_contacts)} θ-contacts")
    
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        print(f"Simulation stopped at time {self.current_time:.2f} seconds")
        print(f"Statistics: {self.stats}")
    
    def tick(self):
        """
        Perform one simulation step (tick).
        
        Returns:
            bool: True if simulation is still running, False otherwise
        """
        if not self.is_running:
            return False
        
        # Check if simulation duration is reached
        if self.current_time >= SIMULATION_DURATION:
            self.stop()
            return False
        
        # 1. Update ocean currents (handled by OceanArea)
        
        # 2. Update ε-node positions based on currents
        for node in self.epsilon_nodes:
            node.update(self.current_time, self.ocean_area)
        
        # 3. Update θ-contact positions
        for contact in self.theta_contacts:
            contact.update(self.current_time, self.ocean_area)
        
        # 4. Detect θ-contacts
        for epsilon_node in self.epsilon_nodes:
            for contact in self.theta_contacts:
                if epsilon_node.detect_contact(contact, self.current_time):
                    self.stats["detections"] += 1
        
        # 5. Process communications
        self.rpr.process_communications(
            self.epsilon_nodes, self.sigma_nodes, self.current_time
        )
        
        # 6. Collect data from σ-nodes
        self.dob.collect_from_sigma_nodes(self.sigma_nodes)
        
        # 7. Update statistics
        self.stats["messages_delivered"] = len(self.rpr.delivered_messages)
        
        # 8. Update visualization if callback is registered
        if self.visualization_callback:
            self.visualization_callback(self)
        
        # Increment time - apply simulation speed factor
        self.current_time += TIME_STEP * SIMULATION_SPEED
        
        return True
    
    def run_simulation(self, duration=None):
        """
        Run the simulation for a specified duration or until completion.
        
        Args:
            duration (float, optional): Duration to run in seconds
            
        Returns:
            dict: Simulation statistics
        """
        if duration is None:
            duration = SIMULATION_DURATION
        
        start_real_time = time.time()
        target_end_time = self.current_time + duration
        
        self.start()
        
        while self.is_running and self.current_time < target_end_time:
            self.tick()
        
        end_real_time = time.time()
        elapsed = end_real_time - start_real_time
        
        print(f"Simulation ran for {duration:.2f} simulated seconds in {elapsed:.2f} real seconds")
        print(f"Speed factor: {duration/elapsed:.2f}x real-time")
        
        return self.stats
    
    def get_state_snapshot(self):
        """
        Get a snapshot of the current simulation state.
        
        Returns:
            dict: Dictionary containing simulation state
        """
        return {
            "time": self.current_time,
            "epsilon_nodes": [(node.id, node.position.copy()) for node in self.epsilon_nodes],
            "sigma_nodes": [(node.id, node.position.copy()) for node in self.sigma_nodes],
            "theta_contacts": [(contact.id, contact.type, contact.position.copy()) 
                               for contact in self.theta_contacts],
            "stats": self.stats.copy()
        }
