#!/usr/bin/env python3
"""
Generate simple geometric visualizations showing Delaunay triangulation and Voronoi diagrams.
Simple, clean illustrations for visual intuition - similar to textbook diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi

def create_simple_delaunay():
    """Create simple Delaunay triangulation visualization."""
    # Simple set of 7 points
    points = np.array([
        [2, 2],
        [1, 4], 
        [3, 5],
        [5, 4],
        [4, 2],
        [2.5, 3],
        [3.5, 3.5]
    ])
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot triangulation edges
    for simplex in tri.simplices:
        triangle = points[simplex]
        # Close the triangle
        triangle = np.vstack([triangle, triangle[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=1.5)
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c='black', s=50, zorder=5)
    
    # Clean styling
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(1.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Simple title
    ax.text(3, 0.8, 'Delaunay triangulation\nfor sensor network', 
            ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/adantas/dev/phd/aquascan-gnns/simple_delaunay.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return points

def create_simple_voronoi(points):
    """Create simple Voronoi diagram visualization."""
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot finite Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:  # Finite ridge
            segment = vor.vertices[simplex]
            ax.plot(segment[:, 0], segment[:, 1], 'k-', linewidth=1)
    
    # Plot boundary box for infinite edges
    box_bounds = [0.5, 5.5, 1.5, 5.5]  # [xmin, xmax, ymin, ymax]
    
    # Handle infinite ridges by extending to box boundary
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in simplex:  # Infinite ridge
            # Find the finite vertex
            finite_vertex_idx = [i for i in simplex if i >= 0][0]
            finite_vertex = vor.vertices[finite_vertex_idx]
            
            # Direction from midpoint of the two generating points
            midpoint = points[pointidx].mean(axis=0)
            direction = finite_vertex - midpoint
            direction = direction / np.linalg.norm(direction)
            
            # Extend line to boundary
            if abs(direction[0]) > abs(direction[1]):  # More horizontal
                if direction[0] > 0:  # Going right
                    end_point = [box_bounds[1], finite_vertex[1] + direction[1] * (box_bounds[1] - finite_vertex[0]) / direction[0]]
                else:  # Going left
                    end_point = [box_bounds[0], finite_vertex[1] + direction[1] * (box_bounds[0] - finite_vertex[0]) / direction[0]]
            else:  # More vertical
                if direction[1] > 0:  # Going up
                    end_point = [finite_vertex[0] + direction[0] * (box_bounds[3] - finite_vertex[1]) / direction[1], box_bounds[3]]
                else:  # Going down
                    end_point = [finite_vertex[0] + direction[0] * (box_bounds[2] - finite_vertex[1]) / direction[1], box_bounds[2]]
            
            # Plot the line
            ax.plot([finite_vertex[0], end_point[0]], 
                   [finite_vertex[1], end_point[1]], 'k-', linewidth=1)
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c='black', s=50, zorder=5)
    
    # Clean styling
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(1.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Simple title
    ax.text(3, 0.8, 'Voronoi diagrams\nfor sensor network', 
            ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/adantas/dev/phd/aquascan-gnns/simple_voronoi.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate simple topology visualizations."""
    print("Generating simple topology visualizations...")
    
    # Create Delaunay visualization
    points = create_simple_delaunay()
    print("✓ Generated simple_delaunay.png")
    
    # Create Voronoi visualization
    create_simple_voronoi(points)
    print("✓ Generated simple_voronoi.png")
    
    print("\nSimple visualizations complete!")

if __name__ == "__main__":
    main()
