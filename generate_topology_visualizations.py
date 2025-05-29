#!/usr/bin/env python3
"""
Generate academic-quality visualizations showing Delaunay triangulation and Voronoi diagrams 
used in the Aquascan sensor network topology management system.

Creates two publication-ready figures:
1. Delaunay triangulation for permanent connections
2. Voronoi diagram for intermittent connections

Based on the network topology implementation in aquascan/simulation/network_topology.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.patheffects as path_effects

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 1.5,
    'lines.markersize': 8
})

def generate_realistic_sensor_positions(n_nodes=12, area_size=(20, 15), seed=42):
    """
    Generate realistic sensor node positions for marine environment.
    Based on practical deployment considerations with some clustering and spacing variation.
    """
    np.random.seed(seed)
    
    # Create a mix of clustered and distributed positions
    positions = []
    
    # Group 1: Western cluster (4 nodes)
    cluster1_center = (4, 8)
    for i in range(4):
        angle = (i * 2 * np.pi / 4) + np.random.normal(0, 0.3)
        radius = 2 + np.random.normal(0, 0.5)
        x = cluster1_center[0] + radius * np.cos(angle)
        y = cluster1_center[1] + radius * np.sin(angle)
        positions.append([max(1, min(x, area_size[0]-1)), max(1, min(y, area_size[1]-1))])
    
    # Group 2: Eastern cluster (4 nodes) 
    cluster2_center = (16, 7)
    for i in range(4):
        angle = (i * 2 * np.pi / 4) + np.random.normal(0, 0.3)
        radius = 2.5 + np.random.normal(0, 0.5)
        x = cluster2_center[0] + radius * np.cos(angle)
        y = cluster2_center[1] + radius * np.sin(angle)
        positions.append([max(1, min(x, area_size[0]-1)), max(1, min(y, area_size[1]-1))])
    
    # Group 3: Central and edge nodes (4 nodes)
    central_positions = [
        [10, 12],  # Northern central
        [10, 3],   # Southern central
        [1, 5],    # Western edge
        [19, 10]   # Eastern edge
    ]
    
    for pos in central_positions:
        # Add small random variation
        x = pos[0] + np.random.normal(0, 0.5)
        y = pos[1] + np.random.normal(0, 0.5)
        positions.append([max(1, min(x, area_size[0]-1)), max(1, min(y, area_size[1]-1))])
    
    return np.array(positions)

def create_delaunay_visualization():
    """
    Create visualization showing Delaunay triangulation for permanent sensor network connections.
    """
    # Generate sensor positions
    positions = generate_realistic_sensor_positions()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Compute Delaunay triangulation
    tri = Delaunay(positions)
    
    # Plot deployment area boundary
    area_rect = patches.Rectangle(
        (0, 0), 20, 15,
        linewidth=2, edgecolor='lightgray', facecolor='none',
        linestyle='--', alpha=0.7
    )
    ax.add_patch(area_rect)
    
    # Filter triangulation edges by permanent range (5km)
    permanent_range = 5.0
    permanent_edges = []
    
    for simplex in tri.simplices:
        # For each edge in the triangle
        pairs = [(simplex[0], simplex[1]), 
                 (simplex[1], simplex[2]), 
                 (simplex[0], simplex[2])]
        
        for i, j in pairs:
            # Calculate distance
            dist = np.linalg.norm(positions[i] - positions[j])
            
            # Add if within permanent range
            if dist <= permanent_range:
                permanent_edges.append([positions[i], positions[j]])
    
    # Plot permanent connections (Delaunay edges)
    if permanent_edges:
        lc = LineCollection(permanent_edges, colors='blue', 
                           linewidths=2, alpha=0.8, label='Permanent connections (≤5km)')
        ax.add_collection(lc)
    
    # Plot sensor nodes
    ax.scatter(positions[:, 0], positions[:, 1], 
              c='navy', s=150, alpha=0.9, marker='o', 
              edgecolors='white', linewidths=2, zorder=5,
              label='Sensor nodes (ε-nodes)')
    
    # Add node IDs
    for i, pos in enumerate(positions):
        ax.annotate(f'ε{i+1}', (pos[0], pos[1]),
                   xytext=(0, -25), textcoords='offset points',
                   ha='center', fontsize=10, color='navy', fontweight='bold')
    
    # Add range circles around a few example nodes to show connection criteria
    example_nodes = [0, 5, 8]  # Show range for 3 nodes
    for node_idx in example_nodes:
        range_circle = patches.Circle(
            positions[node_idx], permanent_range,
            linewidth=1.5, edgecolor='red', facecolor='none',
            linestyle=':', alpha=0.6
        )
        ax.add_patch(range_circle)
    
    # Add range circle legend entry
    ax.plot([], [], 'r:', linewidth=1.5, alpha=0.6, label='Connection range (5km)')
    
    # Styling
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 16)
    ax.set_xlabel('Distance along coastline (km)', fontsize=14)
    ax.set_ylabel('Distance from shore (km)', fontsize=14)
    ax.set_title('Delaunay Triangulation for Permanent Sensor Network Connections', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add explanation text
    explanation = ('Delaunay triangulation creates optimal triangular mesh\n'
                  'connecting sensor nodes for permanent network backbone.\n'
                  'Only connections within 5km range are established.')
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/Users/adantas/dev/phd/aquascan-gnns/delaunay_topology.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated delaunay_topology.png")
    return positions

def create_voronoi_visualization(positions):
    """
    Create visualization showing Voronoi diagram for intermittent sensor network connections.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Compute Voronoi diagram
    vor = Voronoi(positions)
    
    # Plot deployment area boundary
    area_rect = patches.Rectangle(
        (0, 0), 20, 15,
        linewidth=2, edgecolor='lightgray', facecolor='none',
        linestyle='--', alpha=0.7
    )
    ax.add_patch(area_rect)
    
    # Plot Voronoi diagram with custom styling
    # Plot finite segments
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in simplex:  # Finite ridge
            segment = vor.vertices[simplex]
            ax.plot(segment[:, 0], segment[:, 1], 'k-', alpha=0.4, linewidth=1)
    
    # Plot infinite segments (rays) - simplified approach
    center = positions.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in simplex:  # Infinite ridge
            simplex_array = np.array(simplex)
            i = simplex_array[simplex_array >= 0][0]  # Finite vertex
            t = positions[pointidx[1]] - positions[pointidx[0]]  # Tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal
            
            midpoint = positions[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 50
            
            # Only plot if within reasonable bounds
            if (-2 < far_point[0] < 22 and -2 < far_point[1] < 17):
                ax.plot([vor.vertices[i, 0], far_point[0]], 
                       [vor.vertices[i, 1], far_point[1]], 'k-', alpha=0.4, linewidth=1)
    
    # Identify and plot some intermittent connections based on Voronoi adjacency
    intermittent_range_min = 5.0
    intermittent_range_max = 10.0
    intermittent_connections = []
    
    # Find Voronoi neighbors (nodes that share an edge)
    for pointidx in vor.ridge_points:
        i, j = pointidx
        dist = np.linalg.norm(positions[i] - positions[j])
        
        # Add as intermittent if in the right range
        if intermittent_range_min < dist <= intermittent_range_max:
            # Add with some probability to avoid overcrowding
            if np.random.random() < 0.7:  # 70% chance
                intermittent_connections.append([positions[i], positions[j]])
    
    # Plot intermittent connections
    if intermittent_connections:
        lc = LineCollection(intermittent_connections, colors='orange', 
                           linewidths=2.5, alpha=0.8, linestyles='dashed',
                           label='Intermittent connections (5-10km)')
        ax.add_collection(lc)
    
    # Plot sensor nodes
    ax.scatter(positions[:, 0], positions[:, 1], 
              c='navy', s=150, alpha=0.9, marker='o', 
              edgecolors='white', linewidths=2, zorder=5,
              label='Sensor nodes (ε-nodes)')
    
    # Add node IDs
    for i, pos in enumerate(positions):
        ax.annotate(f'ε{i+1}', (pos[0], pos[1]),
                   xytext=(0, -25), textcoords='offset points',
                   ha='center', fontsize=10, color='navy', fontweight='bold')
    
    # Add some Voronoi cell highlighting for educational purposes
    # Highlight a few cells to show the concept
    highlight_nodes = [3, 7, 10]  # Highlight these cells
    for node_idx in highlight_nodes:
        # Find the Voronoi cell vertices for this node
        region = vor.regions[vor.point_region[node_idx]]
        if -1 not in region and len(region) > 0:
            polygon_verts = [vor.vertices[i] for i in region]
            if len(polygon_verts) > 2:
                polygon = patches.Polygon(polygon_verts, alpha=0.15, 
                                        facecolor='lightblue', edgecolor='blue',
                                        linewidth=1.5)
                ax.add_patch(polygon)
    
    # Styling
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 16)
    ax.set_xlabel('Distance along coastline (km)', fontsize=14)
    ax.set_ylabel('Distance from shore (km)', fontsize=14)
    ax.set_title('Voronoi Diagram for Intermittent Sensor Network Connections', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add explanation text
    explanation = ('Voronoi cells define natural neighborhoods.\n'
                  'Nodes sharing cell boundaries can form\n'
                  'intermittent connections (5-10km range).')
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
    
    # Create custom legend entries
    legend_elements = [
        plt.scatter([], [], c='navy', s=150, marker='o', 
                   edgecolors='white', linewidths=2, label='Sensor nodes (ε-nodes)'),
        plt.Line2D([0], [0], color='black', alpha=0.4, linewidth=1, 
                   label='Voronoi cell boundaries'),
        plt.Line2D([0], [0], color='orange', linewidth=2.5, linestyle='dashed',
                   alpha=0.8, label='Intermittent connections (5-10km)'),
        patches.Patch(facecolor='lightblue', alpha=0.15, edgecolor='blue',
                     linewidth=1.5, label='Example Voronoi cells')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/Users/adantas/dev/phd/aquascan-gnns/voronoi_topology.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✓ Generated voronoi_topology.png")

def main():
    """Generate both topology visualization images."""
    print("Generating topology visualizations for Aquascan GNN paper...")
    print("Based on network topology implementation in aquascan/simulation/network_topology.py")
    print()
    
    # Generate Delaunay triangulation visualization
    print("Creating Delaunay triangulation visualization...")
    positions = create_delaunay_visualization()
    
    # Generate Voronoi diagram visualization  
    print("Creating Voronoi diagram visualization...")
    create_voronoi_visualization(positions)
    
    print()
    print("✓ Both visualizations generated successfully!")
    print("Files saved:")
    print("  - delaunay_topology.png: Shows permanent connections via Delaunay triangulation")
    print("  - voronoi_topology.png: Shows intermittent connections via Voronoi diagrams")
    print()
    print("These publication-ready figures demonstrate how your Aquascan system uses")
    print("computational geometry for sensor network topology management.")

if __name__ == "__main__":
    main()
