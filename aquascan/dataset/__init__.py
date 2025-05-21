"""
Dataset module for converting simulation data to graph structures.

This module provides utilities for building graph datasets from raw simulation
output files, suitable for training graph neural networks.
"""

from aquascan.dataset.build_graphs import (
    h5_to_graphs,
    create_graph_from_window,
    save_split,
    validate_graphs
)

__all__ = [
    'h5_to_graphs',
    'create_graph_from_window',
    'save_split',
    'validate_graphs'
]