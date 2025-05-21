"""
Aquascan Sensors Module

This module defines the sensing capabilities of ε-nodes.
Responsibilities:
- Handle sensor configurations (sonar, hydrophones, cameras)
- Define detection ranges and capabilities
- Implement sensor noise and variability
- Process detections of marine entities
"""

import numpy as np
from aquascan.config.simulation_config import DETECTION_RADIUS


class SensorArray:
    """
    Models the sensor array on an ε-node.
    
    Includes simulated sonar, hydrophones, and cameras.
    """
    
    def __init__(self, node):
        """
        Initialize the sensor array.
        
        Args:
            node (EpsilonNode): The node this sensor array belongs to
        """
        self.node = node
        self.detection_radius = DETECTION_RADIUS
        self.last_detection_time = 0
        
        # Sensor noise parameters
        self.position_error = 0.01  # km (10m position error)
        self.volume_error = 0.1  # Relative error in volume estimation
    
    def detect_entities(self, entities, current_time):
        """
        Detect entities within sensor range.
        
        Args:
            entities (list): List of marine entities to check for detection
            current_time (float): Current simulation time
            
        Returns:
            list: List of detected entities with detection data
        """
        detected = []
        
        for entity in entities:
            distance = np.linalg.norm(self.node.position - entity.position)
            
            if distance <= self.detection_radius:
                # Entity is within detection range
                
                # Add noise to position
                position_noise = np.random.normal(0, self.position_error, 2)
                noisy_position = entity.position + position_noise
                
                # Add noise to volume
                volume_noise = 1.0 + np.random.normal(0, self.volume_error)
                noisy_volume = entity.volume * volume_noise
                
                # Create detection data
                detection = {
                    "entity_id": entity.id,
                    "entity_type": entity.type,
                    "position": noisy_position,
                    "distance": distance,
                    "estimated_volume": max(0, noisy_volume),
                    "time": current_time
                }
                
                detected.append(detection)
                self.last_detection_time = current_time
        
        return detected
