"""
Communication Module

This module implements the communication protocols between nodes in the Aquascan simulation.
Responsibilities:
- Implement the Reliable Proximity Relay (RPR) protocol
- Define the Spatiotemporal Contact Volume (SCV) data structure
- Handle data transmission between ε-nodes
- Manage relay of information to σ-nodes
- Implement network delays and constraints
"""

import numpy as np
import time
from aquascan.config.simulation_config import MAX_COMM_RANGE, OPTIMAL_COMM_RANGE


class ReliableProximityRelay:
    """
    Implements the Reliable Proximity Relay (RPR) protocol.
    
    RPR handles communication between ε-nodes and relaying data to σ-nodes.
    It operates within a bounded range and includes network delays.
    """
    
    def __init__(self, ocean_area):
        """
        Initialize the RPR protocol handler.
        
        Args:
            ocean_area (OceanArea): Reference to the ocean area object
        """
        self.ocean_area = ocean_area
        self.message_queue = []  # Queue of messages to be delivered
        self.delivered_messages = set()  # Set of message IDs that have been delivered
    
    def transmit_scv(self, source_node, scv, current_time):
        """
        Transmit a Spatiotemporal Contact Volume (SCV) from an ε-node.
        
        Args:
            source_node (EpsilonNode): Node that generated the SCV
            scv (dict): SCV data structure
            current_time (float): Current simulation time
            
        Returns:
            str: Message ID of the transmitted SCV
        """
        # Generate a unique message ID
        message_id = f"msg-{current_time}-{source_node.id}-{len(self.message_queue)}"
        
        # Create a message object
        message = {
            "id": message_id,
            "source": source_node.id,
            "data": scv,
            "creation_time": current_time,
            "delivery_time": None,  # Will be set when delivered
            "hops": 0,
            "max_hops": 10,  # Maximum number of hops allowed
            "delivered_to": set()  # Set of nodes that have received this message
        }
        
        # Add to message queue
        self.message_queue.append(message)
        
        return message_id
    
    def process_communications(self, nodes, sigma_nodes, current_time):
        """
        Process all pending communications for the current time step.
        
        This function simulates the propagation of messages through the network,
        including node-to-node communication and relaying to σ-nodes.
        
        Args:
            nodes (list): List of all ε-nodes
            sigma_nodes (list): List of all σ-nodes
            current_time (float): Current simulation time
        """
        # Dictionary to map node IDs to actual node objects
        node_map = {node.id: node for node in nodes}
        sigma_map = {node.id: node for node in sigma_nodes}
        
        # Dictionary to keep track of which messages each node has
        node_messages = {node.id: set() for node in nodes}
        
        # First, process existing messages in the queue
        remaining_messages = []
        
        for message in self.message_queue:
            # Skip messages that have reached max hops
            if message["hops"] >= message["max_hops"]:
                continue
                
            # Try to deliver to σ-nodes first (priority)
            delivered_to_sigma = False
            
            for sigma_id, sigma_node in sigma_map.items():
                source_node = node_map.get(message["source"])
                if source_node and self._can_communicate(source_node, sigma_node):
                    # Message delivered to σ-node
                    sigma_node.buffer.append(message["data"])
                    message["delivery_time"] = current_time
                    self.delivered_messages.add(message["id"])
                    delivered_to_sigma = True
                    break
            
            if delivered_to_sigma:
                continue
            
            # If not delivered to a σ-node, propagate to neighboring ε-nodes
            message["hops"] += 1
            propagated = False
            
            for node_id, node in node_map.items():
                # Skip the source node
                if node_id == message["source"]:
                    continue
                
                # Skip nodes that already have this message
                if node_id in message["delivered_to"]:
                    continue
                
                # Check if nodes can communicate
                source_node = node_map.get(message["source"])
                if source_node and self._can_communicate(source_node, node):
                    # Propagate message to this node
                    message["delivered_to"].add(node_id)
                    node_messages[node_id].add(message["id"])
                    propagated = True
            
            # Keep in queue if still propagating
            if propagated:
                remaining_messages.append(message)
        
        # Update the message queue
        self.message_queue = remaining_messages
        
        # Now, generate new messages from node DOBs
        for node in nodes:
            if node.dob:
                # Try to send to σ-nodes first
                for sigma_node in sigma_nodes:
                    if self._can_communicate(node, sigma_node):
                        sigma_node.collect_data(node)
                        break
    
    def _can_communicate(self, node1, node2):
        """
        Check if two nodes can communicate based on distance.
        
        Args:
            node1, node2: The two nodes to check communication between
            
        Returns:
            bool: True if communication is possible, False otherwise
        """
        distance = np.linalg.norm(node1.position - node2.position)
        
        # Communication is possible if within max range
        if distance <= MAX_COMM_RANGE:
            # Calculate communication reliability based on distance
            reliability = 1.0 - (distance / MAX_COMM_RANGE)
            
            # More reliable within optimal range
            if distance <= OPTIMAL_COMM_RANGE:
                reliability = max(0.9, reliability)
            
            # Simulate packet loss
            return np.random.random() < reliability
        
        return False


class SpatiotemporalContactVolume:
    """
    Factory class for creating Spatiotemporal Contact Volume (SCV) data structures.
    
    SCV is a data structure that captures detected entity information including
    position, volume, and entity type.
    """
    
    @staticmethod
    def create(epsilon_node, theta_contact, timestamp):
        """
        Create a new SCV data structure.
        
        Args:
            epsilon_node (EpsilonNode): Node that detected the contact
            theta_contact (ThetaContact): The detected marine entity
            timestamp (float): Time of detection
            
        Returns:
            dict: SCV data structure
        """
        return {
            "epsilon_id": epsilon_node.id,
            "timestamp": timestamp,
            "theta_id": theta_contact.id,
            "position": theta_contact.position.copy(),
            "estimated_volume": theta_contact.volume,
            "entity_type": theta_contact.species_name
        }


class DistributedObservationBuffer:
    """
    Manages the Distributed Observation Buffer (DOB) system.
    
    The DOB is responsible for storing observations and node metadata
    throughout the network.
    """
    
    def __init__(self):
        """Initialize the DOB system."""
        self.global_buffer = []  # Central storage for all data
    
    def collect_from_sigma_nodes(self, sigma_nodes):
        """
        Collect data from all σ-nodes into the global buffer.
        
        Args:
            sigma_nodes (list): List of all σ-nodes
        """
        for node in sigma_nodes:
            if node.buffer:
                self.global_buffer.extend(node.buffer)
                node.buffer = []  # Clear node buffer after collection
    
    def get_observations(self, start_time=None, end_time=None, entity_type=None):
        """
        Retrieve filtered observations from the global buffer.
        
        Args:
            start_time (float, optional): Filter by minimum timestamp
            end_time (float, optional): Filter by maximum timestamp
            entity_type (str, optional): Filter by entity type
            
        Returns:
            list: Filtered observations
        """
        filtered = self.global_buffer
        
        if start_time is not None:
            filtered = [obs for obs in filtered if obs["timestamp"] >= start_time]
        
        if end_time is not None:
            filtered = [obs for obs in filtered if obs["timestamp"] <= end_time]
        
        if entity_type is not None:
            filtered = [obs for obs in filtered if obs["entity_type"] == entity_type]
        
        return filtered
