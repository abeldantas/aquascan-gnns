#!/usr/bin/env python3
"""
Generate clean topology visualizations for paper showing Delaunay triangulation 
and Voronoi diagrams as used in Aquascan marine sensor networks.

Creates simple, textbook-style illustrations with ~12 nodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi
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
    
    # Create figure with clean styling
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Add subtle grid background
    ax.grid(True, linestyle=':', alpha=0.2, color='gray')
    
    # Define clipping boundaries
    x_min, x_max = -1, 8
    y_min, y_max = -1, 8
    
    # Plot finite Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:  # Finite ridge
            segment = vor.vertices[simplex]
            ax.plot(segment[:, 0], segment[:, 1], 'k-', linewidth=1.2)
    
    # Handle infinite ridges
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in simplex:  # Infinite ridge
            # Find the finite vertex
            i = simplex[1] if simplex[0] == -1 else simplex[0]
            if i == -1:
                continue
                
            t = points[pointidx[1]] - points[pointidx[0]]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal
            
            midpoint = points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + n * 100 * np.sign(np.dot(n, vor.vertices[i] - midpoint))
            
            # Clip the line to the bounding box
            line_points = [vor.vertices[i], far_point]
            
            # Simple clipping (could be more sophisticated)
            x1, y1 = line_points[0]
            x2, y2 = line_points[1]
            
            # Clip to boundaries
            if x2 > x_max:
                y2 = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x2 = x_max
            elif x2 < x_min:
                y2 = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x2 = x_min
                
            if y2 > y_max:
                x2 = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y2 = y_max
            elif y2 < y_min:
                x2 = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y2 = y_min
            
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.2)
    
    # Plot sensor nodes
    ax.scatter(points[:, 0], points[:, 1], c='black', s=80, zorder=5)
    
    # Shade one or two Voronoi cells to highlight coverage areas
    highlight_nodes = [4, 7]  # Central nodes
    for node_idx in highlight_nodes:
        # Find the Voronoi region for this point
        region_idx = vor.point_region[node_idx]
        if region_idx >= 0:
            region = vor.regions[region_idx]
            if len(region) > 0 and -1 not in region:
                polygon = [vor.vertices[i] for i in region]
                poly = patches.Polygon(polygon, alpha=0.15, facecolor='gray', 
                                     edgecolor='none')
                ax.add_patch(poly)
    
    # Add sensor labels
    for i, point in enumerate(points):
        ax.text(point[0] + 0.1, point[1] + 0.1, f'{i+1}', 
                fontsize=8, ha='left', va='bottom')
    
    # Set limits and aspect
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    # Remove axis but keep frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Add title
    ax.text(3.5, -1.5, 'Voronoi diagrams', 
            ha='center', fontsize=14, style='italic')
    ax.text(3.5, -2.0, 'for marine sensor network', 
            ha='center', fontsize=14, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Generate paper-ready topology visualizations."""
    print("Generating paper topology figures...")
    
    # Create Delaunay visualization
    delaunay_path = '/Users/adantas/dev/phd/aquascan-gnns/paper_delaunay.png'
    points = create_delaunay_figure(delaunay_path)
    print(f"✓ Generated {delaunay_path}")
    
    # Create Voronoi visualization with same points
    voronoi_path = '/Users/adantas/dev/phd/aquascan-gnns/paper_voronoi.png'
    create_voronoi_figure(points, voronoi_path)
    print(f"✓ Generated {voronoi_path}")
    
    print("\nPaper figures complete!")


if __name__ == "__main__":
    main()
