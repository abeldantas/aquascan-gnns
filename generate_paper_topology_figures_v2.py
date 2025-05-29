#!/usr/bin/env python3
"""
Generate clean topology visualizations for paper showing Delaunay triangulation 
and Voronoi diagrams as used in Aquascan marine sensor networks.

Creates simple, textbook-style illustrations with ~12 nodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.patches as patches


def generate_marine_sensor_positions(n_sensors=12, seed=42):
    """Generate sensor positions simulating a marine deployment pattern."""
    np.random.seed(seed)
    
    # Create a more realistic marine sensor distribution
    # Mix of grid-like placement with some natural variation
    positions = []
    
    # Core grid structure (3x3)
    grid_points = 9
    grid_size = 3
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * 2 + 1 + np.random.normal(0, 0.3)
            y = j * 2 + 1 + np.random.normal(0, 0.3)
            positions.append([x, y])
    
    # Add a few outer sensors
    outer_points = n_sensors - grid_points
    angles = np.linspace(0, 2*np.pi, outer_points, endpoint=False)
    for angle in angles:
        r = 4.5 + np.random.normal(0, 0.3)
        x = 3.5 + r * np.cos(angle)
        y = 3.5 + r * np.sin(angle)
        positions.append([x, y])
    
    return np.array(positions)


def create_delaunay_figure(save_path):
    """Create Delaunay triangulation visualization for marine sensor network."""
    # Generate sensor positions
    points = generate_marine_sensor_positions(12)
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot triangulation edges, filtering out long connections
    for simplex in tri.simplices:
        triangle = points[simplex]
        
        # Check each edge in the triangle
        edges = [(0, 1), (1, 2), (0, 2)]
        for i, j in edges:
            p1, p2 = triangle[i], triangle[j]
            dist = np.linalg.norm(p1 - p2)
            
            # Skip very long edges (threshold based on typical edge length)
            if dist < 4.5:  # Adjust threshold as needed
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.5)
    
    # Plot sensor nodes
    ax.scatter(points[:, 0], points[:, 1], c='black', s=80, zorder=5)
    
    # Set limits and aspect
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    
    # Remove all axes and frame
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()
    
    return points


def create_voronoi_figure(points, save_path):
    """Create Voronoi diagram visualization for marine sensor network."""
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Use the built-in voronoi plotting function as a base
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', 
                    line_width=1.2, line_alpha=1.0, point_size=0)
    
    # Plot sensor nodes on top
    ax.scatter(points[:, 0], points[:, 1], c='black', s=80, zorder=5)
    
    # Set limits and aspect
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    
    # Remove all axes and frame
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


def main():
    """Generate paper-ready topology visualizations."""
    print("Generating paper topology figures...")
    
    # Create Delaunay visualization
    delaunay_path = '/Users/adantas/dev/phd/aquascan-gnns/paper_delaunay_v2.png'
    points = create_delaunay_figure(delaunay_path)
    print(f"✓ Generated {delaunay_path}")
    
    # Create Voronoi visualization with same points
    voronoi_path = '/Users/adantas/dev/phd/aquascan-gnns/paper_voronoi_v2.png'
    create_voronoi_figure(points, voronoi_path)
    print(f"✓ Generated {voronoi_path}")
    
    print("\nPaper figures complete!")


if __name__ == "__main__":
    main()
