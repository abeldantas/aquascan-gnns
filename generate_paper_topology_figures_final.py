#!/usr/bin/env python3
"""
Generate clean topology visualizations for paper showing Delaunay triangulation 
and Voronoi diagrams as used in Aquascan marine sensor networks.

Creates simple, textbook-style illustrations with ~12 nodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi
from matplotlib.collections import LineCollection


def generate_marine_sensor_positions(n_sensors=12, seed=42):
    """Generate sensor positions simulating a marine deployment pattern."""
    np.random.seed(seed)
    
    # Create a more realistic marine sensor distribution
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
            
            # Skip very long edges (more restrictive threshold)
            if dist < 3.0:  # Further reduced to remove all long edges
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.5)
    
    # Plot sensor nodes
    ax.scatter(points[:, 0], points[:, 1], c='black', s=80, zorder=5)
    
    # Set limits and aspect - ensure all nodes and edges are visible
    ax.set_xlim(-1.5, 8.5)
    ax.set_ylim(-1.5, 8.5)
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
    
    # Define clipping box - ensure all nodes are visible
    x_min, x_max = -1.5, 8.5
    y_min, y_max = -1.5, 8.5
    
    # Plot only finite Voronoi edges
    segments = []
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):  # Both vertices are finite
            segment = vor.vertices[simplex]
            segments.append(segment)
    
    # Add finite segments as solid lines
    if segments:
        lc = LineCollection(segments, colors='black', linewidths=1.2)
        ax.add_collection(lc)
    
    # Handle infinite ridges more carefully
    center = points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):  # One vertex is at infinity
            i = simplex[simplex >= 0][0]  # Finite vertex
            
            # Calculate direction to extend the ridge
            t = points[pointidx[1]] - points[pointidx[0]]  # Tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # Normal
            
            midpoint = points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            
            # Extend ridge to a far point
            far_point = vor.vertices[i] + direction * 20
            
            # Clip line to box
            line = clip_line_to_box(vor.vertices[i], far_point, 
                                  x_min, x_max, y_min, y_max)
            if line is not None:
                ax.plot([line[0][0], line[1][0]], 
                       [line[0][1], line[1][1]], 'k-', linewidth=1.2)
    
    # Plot sensor nodes on top
    ax.scatter(points[:, 0], points[:, 1], c='black', s=80, zorder=5)
    
    # Set limits and aspect
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    # Remove all axes and frame
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()


def clip_line_to_box(p1, p2, x_min, x_max, y_min, y_max):
    """Clip a line segment to a rectangular box using Cohen-Sutherland algorithm."""
    def compute_code(x, y):
        code = 0
        if x < x_min: code |= 1  # left
        elif x > x_max: code |= 2  # right
        if y < y_min: code |= 4  # bottom
        elif y > y_max: code |= 8  # top
        return code
    
    x1, y1 = p1
    x2, y2 = p2
    code1 = compute_code(x1, y1)
    code2 = compute_code(x2, y2)
    
    while True:
        if code1 == 0 and code2 == 0:  # Both inside
            return [[x1, y1], [x2, y2]]
        elif code1 & code2 != 0:  # Both outside same region
            return None
        
        # At least one endpoint is outside
        code = code1 if code1 != 0 else code2
        
        # Find intersection with box
        if code & 8:  # top
            x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
            y = y_max
        elif code & 4:  # bottom
            x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
            y = y_min
        elif code & 2:  # right
            y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
            x = x_max
        elif code & 1:  # left
            y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
            x = x_min
        
        # Update the point outside the box
        if code == code1:
            x1, y1 = x, y
            code1 = compute_code(x1, y1)
        else:
            x2, y2 = x, y
            code2 = compute_code(x2, y2)


def main():
    """Generate paper-ready topology visualizations."""
    print("Generating paper topology figures...")
    
    # Create Delaunay visualization
    delaunay_path = '/Users/adantas/dev/phd/aquascan-gnns/paper_delaunay_final.png'
    points = create_delaunay_figure(delaunay_path)
    print(f"✓ Generated {delaunay_path}")
    
    # Create Voronoi visualization with same points
    voronoi_path = '/Users/adantas/dev/phd/aquascan-gnns/paper_voronoi_final.png'
    create_voronoi_figure(points, voronoi_path)
    print(f"✓ Generated {voronoi_path}")
    
    print("\nPaper figures complete!")


if __name__ == "__main__":
    main()
