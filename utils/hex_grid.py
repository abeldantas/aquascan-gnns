"""
Utility module for creating and working with hexagonal grid layouts.

This module provides functions for:
- Creating hexagonal grid coordinates
- Converting between hexagonal and Cartesian coordinates
- Calculating distances and neighbors in a hexagonal grid
- Generating initial deployment positions for sensors
"""

import numpy as np


def generate_hex_grid(width, height, spacing):
    """
    Generate a hexagonal grid of points within the specified dimensions.
    
    Args:
        width (float): Width of the area in km
        height (float): Height of the area in km
        spacing (float): Horizontal spacing between points in km
        
    Returns:
        np.ndarray: Array of shape (n, 2) containing (x, y) coordinates
    """
    # Calculate vertical spacing for hexagonal grid
    vert_spacing = (3**0.5/2) * spacing
    
    # Calculate number of points in each dimension
    num_cols = int(width / spacing) + 1
    num_rows = int(height / vert_spacing) + 1
    
    # Create coordinate arrays
    coords = []
    for row in range(num_rows):
        # Shift every other row by half a spacing
        x_offset = (row % 2) * (spacing / 2)
        for col in range(num_cols):
            x = col * spacing + x_offset
            y = row * vert_spacing
            
            # Only include points within the specified dimensions
            if x < width and y < height:
                coords.append([x, y])
    
    return np.array(coords)


def hex_distance(p1, p2):
    """
    Calculate distance between two points in the hex grid.
    
    Args:
        p1, p2: Coordinates of points as (x, y) tuples or arrays
        
    Returns:
        float: Euclidean distance between the points
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def get_neighbors(point, all_points, max_distance):
    """
    Find all neighbors of a point within a given distance.
    
    Args:
        point: (x, y) coordinates of the reference point
        all_points: np.ndarray of shape (n, 2) containing all point coordinates
        max_distance: Maximum distance to consider for neighbors
        
    Returns:
        list: Indices of neighboring points
    """
    distances = np.sqrt(np.sum((all_points - point)**2, axis=1))
    return np.where(distances <= max_distance)[0]


def get_deployment_positions(width, height, resolution):
    """
    Generate deployment positions for a sensor network with the given resolution.
    
    Args:
        width (float): Width of the area in km
        height (float): Height of the area in km
        resolution (float): Spacing between sensors in km
        
    Returns:
        np.ndarray: Array of shape (n, 2) containing (x, y) coordinates for sensors
    """
    return generate_hex_grid(width, height, resolution)
