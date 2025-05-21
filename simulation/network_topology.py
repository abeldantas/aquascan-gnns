"""
Network Topology Module

This module defines different network topology strategies for connecting ε-nodes.
Each topology class is responsible for creating and maintaining network connections
while respecting node constraints (maximum connections, range limits, etc.).

Available topology strategies:
- DelaunayVoronoiTopology: Uses Delaunay triangulation for initial mesh and
  Voronoi diagrams for intermittent connection updates
"""

import numpy as np
from scipy.spatial import Delaunay, Voronoi


class NetworkTopology:
    """Base class for network topology strategies."""
    
    def __init__(self, min_connections=3, max_connections=5, permanent_range=5.0, intermittent_range=10.0, 
                 intermittent_update_interval=7200):
        """
        Initialize the network topology manager.
        
        Args:
            min_connections: Minimum number of connections per node
            max_connections: Maximum number of connections per node
            permanent_range: Maximum distance (km) for permanent connections
            intermittent_range: Maximum distance (km) for intermittent connections
            intermittent_update_interval: Time (s) between intermittent connection updates
        """
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.permanent_range = permanent_range
        self.intermittent_range = intermittent_range
        self.intermittent_update_interval = intermittent_update_interval
        
        # Connection storage
        self.permanent_connections = set()
        self.intermittent_connections = set()
        self.last_intermittent_update = 0
        
        # Initialization flag
        self.initialized = False
    
    def initialize(self, nodes, current_time=0):
        """
        Initialize the network topology.
        
        Args:
            nodes: List of nodes to connect
            current_time: Current simulation time
        """
        self.last_intermittent_update = current_time
        self.initialized = True
    
    def update_connections(self, nodes, current_time):
        """
        Update network connections based on current node positions.
        
        Args:
            nodes: List of nodes to connect
            current_time: Current simulation time
            
        Returns:
            dict: Updated connections {'permanent': set(), 'intermittent': set()}
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize(nodes, current_time)
            
        # Check if intermittent connections need updating
        update_intermittent = False
        if current_time - self.last_intermittent_update >= self.intermittent_update_interval:
            self.intermittent_connections.clear()
            self.last_intermittent_update = current_time
            update_intermittent = True
            
        return {
            'permanent': self.permanent_connections,
            'intermittent': self.intermittent_connections
        }
    
    def get_connections(self):
        """
        Get the current network connections.
        
        Returns:
            dict: Current connections {'permanent': set(), 'intermittent': set()}
        """
        return {
            'permanent': self.permanent_connections,
            'intermittent': self.intermittent_connections
        }


class DelaunayVoronoiTopology(NetworkTopology):
    """
    Network topology strategy using Delaunay triangulation for initial connections
    and Voronoi diagrams for intermittent connection updates.
    
    This creates a mathematically optimal mesh network that is resilient to node failures
    while still respecting the maximum connection constraint per node.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the Delaunay-Voronoi topology."""
        super().__init__(*args, **kwargs)
        self.major_recalc_interval = 3600 * 6  # 6 hours
        self.last_major_recalc = 0
    
    def initialize(self, nodes, current_time=0):
        """Initialize the network with Delaunay triangulation."""
        super().initialize(nodes, current_time)
        self.last_major_recalc = current_time
        
        # Initial Delaunay triangulation
        self._calculate_delaunay_connections(nodes)
    
    def update_connections(self, nodes, current_time):
        """
        Update network connections using Delaunay triangulation for major updates
        and Voronoi diagrams for intermittent connections.
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize(nodes, current_time)
        
        # Major recalculation with Delaunay triangulation
        if current_time - self.last_major_recalc >= self.major_recalc_interval:
            self.permanent_connections.clear()
            self._calculate_delaunay_connections(nodes)
            self.last_major_recalc = current_time
        
        # Intermittent connections update with Voronoi diagram
        update_intermittent = False
        if current_time - self.last_intermittent_update >= self.intermittent_update_interval:
            self.intermittent_connections.clear()
            self.last_intermittent_update = current_time
            update_intermittent = True
            
            if update_intermittent:
                self._calculate_voronoi_connections(nodes)
        
        return self.get_connections()
    
    def _calculate_delaunay_connections(self, nodes):
        """
        Calculate connections using Delaunay triangulation.
        
        This creates an optimal triangular mesh where each node connects to its
        natural neighbors while respecting the connection constraints.
        """
        # Extract node positions and create ID lookup
        node_positions = np.array([node.position for node in nodes])
        node_id_lookup = {i: node.id for i, node in enumerate(nodes)}
        position_lookup = {node.id: i for i, node in enumerate(nodes)}
        
        # Skip if not enough nodes for triangulation
        if len(node_positions) < 4:
            return
        
        # Create Delaunay triangulation
        try:
            tri = Delaunay(node_positions)
        except Exception as e:
            print(f"Error in Delaunay triangulation: {e}")
            return
        
        # Track connections per node
        node_connections = {node.id: [] for node in nodes}
        
        # Process each simplex (triangle) in the triangulation
        for simplex in tri.simplices:
            # For each edge in the triangle
            pairs = [(simplex[0], simplex[1]), 
                     (simplex[1], simplex[2]), 
                     (simplex[0], simplex[2])]
            
            for i, j in pairs:
                # Get node IDs
                id1, id2 = node_id_lookup[i], node_id_lookup[j]
                
                # Skip if either node already at max connections
                if len(node_connections[id1]) >= self.max_connections or \
                   len(node_connections[id2]) >= self.max_connections:
                    continue
                
                # Calculate distance
                dist = np.linalg.norm(node_positions[i] - node_positions[j])
                
                # Add as permanent connection if within range
                if dist <= self.permanent_range:
                    # Ensure consistent ordering
                    connection = (min(id1, id2), max(id1, id2))
                    self.permanent_connections.add(connection)
                    node_connections[id1].append(id2)
                    node_connections[id2].append(id1)
        
        # Check if all nodes have at least min_connections
        # Add additional connections if needed
        self._ensure_min_connections(nodes, node_connections, node_positions, node_id_lookup, position_lookup)
        
        # Verify that the network is fully connected
        self._ensure_full_connectivity(nodes, node_connections, node_positions, node_id_lookup, position_lookup)
        
    def _ensure_min_connections(self, nodes, node_connections, node_positions, node_id_lookup, position_lookup):
        """Ensure all nodes have at least the minimum number of connections."""
        # Find nodes with fewer than min_connections
        under_connected = [node.id for node in nodes 
                          if len(node_connections[node.id]) < self.min_connections]
        
        # For each under-connected node
        for node_id in under_connected:
            # Number of connections needed
            needed = self.min_connections - len(node_connections[node_id])
            
            if needed <= 0:
                continue
                
            # Get node position index
            pos_idx = position_lookup[node_id]
            
            # Calculate distances to all other nodes
            distances = []
            for i, other_node in enumerate(nodes):
                if other_node.id == node_id or other_node.id in node_connections[node_id]:
                    continue
                    
                # Skip if other node already at max connections
                if len(node_connections[other_node.id]) >= self.max_connections:
                    continue
                    
                # Calculate distance
                dist = np.linalg.norm(node_positions[pos_idx] - node_positions[position_lookup[other_node.id]])
                if dist <= self.permanent_range:
                    distances.append((other_node.id, dist))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Add needed connections
            for other_id, dist in distances[:needed]:
                connection = (min(node_id, other_id), max(node_id, other_id))
                self.permanent_connections.add(connection)
                node_connections[node_id].append(other_id)
                node_connections[other_id].append(node_id)
                
                # Check if we have enough connections
                if len(node_connections[node_id]) >= self.min_connections:
                    break
    
    def _ensure_full_connectivity(self, nodes, node_connections, node_positions, node_id_lookup, position_lookup):
        """
        Ensure the network is fully connected by adding bridges between disconnected components.
        Uses a breadth-first search to identify components and then connects them.
        """
        # Build adjacency list for graph connectivity analysis
        adjacency = {node.id: set(node_connections[node.id]) for node in nodes}
        
        # Find connected components using BFS
        visited = set()
        components = []
        
        # For each unvisited node, find its connected component
        for node in nodes:
            if node.id in visited:
                continue
                
            # New component
            component = set()
            queue = [node.id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                    
                visited.add(current)
                component.add(current)
                
                # Add neighbors to queue
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        # If only one component, the network is fully connected
        if len(components) <= 1:
            return
            
        print(f"Network has {len(components)} disconnected components. Adding bridges.")
        
        # Connect components by adding bridges
        for i in range(len(components) - 1):
            # Find closest nodes between components i and i+1
            comp1 = components[i]
            comp2 = components[i+1]
            
            bridge = None
            min_dist = float('inf')
            
            for id1 in comp1:
                for id2 in comp2:
                    # Calculate distance
                    pos1 = node_positions[position_lookup[id1]]
                    pos2 = node_positions[position_lookup[id2]]
                    dist = np.linalg.norm(pos1 - pos2)
                    
                    # Track minimum distance bridge
                    if dist < min_dist:
                        min_dist = dist
                        bridge = (id1, id2)
            
            if bridge:
                id1, id2 = bridge
                
                # Check if the bridge length is within acceptable limits
                # For bridges, we can use the intermittent_range as the maximum allowable distance
                if min_dist <= self.intermittent_range:
                    # Add bridge connection
                    connection = (min(id1, id2), max(id1, id2))
                    
                    # Decide if permanent or intermittent based on distance
                    if min_dist <= self.permanent_range:
                        self.permanent_connections.add(connection)
                        print(f"Added permanent bridge between components: {id1} - {id2} (dist: {min_dist:.2f}km)")
                    else:
                        self.intermittent_connections.add(connection)
                        print(f"Added intermittent bridge between components: {id1} - {id2} (dist: {min_dist:.2f}km)")
                    
                    # Update adjacency
                    adjacency[id1].add(id2)
                    adjacency[id2].add(id1)
                    
                    # Merge components in our tracking
                    components[i+1] = components[i+1].union(components[i])
                else:
                    print(f"Warning: Cannot connect components - minimum distance ({min_dist:.2f}km) exceeds maximum range ({self.intermittent_range}km)")
        
        # Validate network is now connected
        visited = set()
        queue = [nodes[0].id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            
            # Add neighbors to queue
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        if len(visited) == len(nodes):
            print("Network is now fully connected.")
        else:
            print(f"Warning: Network still has disconnected nodes: {len(nodes) - len(visited)} nodes unreachable.")
            # Could add more aggressive connection strategy here if needed
    
    def _calculate_voronoi_connections(self, nodes):
        """
        Calculate intermittent connections using Voronoi diagram.
        
        This identifies natural neighbor relationships based on shared Voronoi edges,
        creating intermittent connections where appropriate.
        """
        # Extract node positions and create ID lookup
        node_positions = np.array([node.position for node in nodes])
        node_id_lookup = {i: node.id for i, node in enumerate(nodes)}
        
        # Skip if not enough nodes for Voronoi diagram
        if len(node_positions) < 3:
            return
        
        # Get current connections from permanent set
        current_connections = {node.id: [] for node in nodes}
        for id1, id2 in self.permanent_connections:
            current_connections[id1].append(id2)
            current_connections[id2].append(id1)
        
        # Create Voronoi diagram
        try:
            vor = Voronoi(node_positions)
        except Exception as e:
            print(f"Error in Voronoi diagram: {e}")
            return
        
        # Process Voronoi ridges (edges between cells)
        if hasattr(vor, 'ridge_points') and hasattr(vor, 'ridge_vertices'):
            for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
                if -1 not in ridge_vertices:  # Skip ridges extending to infinity
                    i, j = ridge_points
                    id1, id2 = node_id_lookup[i], node_id_lookup[j]
                    
                    # Skip if already connected or either has max connections
                    if id2 in current_connections[id1] or \
                       len(current_connections[id1]) >= self.max_connections or \
                       len(current_connections[id2]) >= self.max_connections:
                        continue
                    
                    # Calculate distance
                    dist = np.linalg.norm(node_positions[i] - node_positions[j])
                    
                    # Add as intermittent if in range with 50% chance
                    if self.permanent_range < dist <= self.intermittent_range and np.random.random() < 0.5:
                        connection = (min(id1, id2), max(id1, id2))
                        self.intermittent_connections.add(connection)
                        current_connections[id1].append(id2)
                        current_connections[id2].append(id1)