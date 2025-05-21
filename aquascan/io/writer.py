"""
SnapshotWriter - HDF5 persistence for Aquascan simulation snapshots

This module provides a writer class for serializing simulation state to HDF5,
enabling efficient storage and retrieval of tick-level graph snapshots.
The resulting files can be streamed or windowed without re-running the simulation,
supporting model training, visualization, and analysis downstream.

File structure:
- /nodes: Structured array of all nodes across all ticks (chunked on t)
- /edges: Structured array of all edges across all ticks (chunked on t)
- /globals: JSON string with simulation metadata (config, seed, etc.)

Usage:
    writer = SnapshotWriter(path, metadata, est_ticks)
    for t in range(ticks):
        sim.tick()
        nodes, edges = sim.export_snapshot(t)
        writer.append(t, nodes, edges)
    writer.close()
"""

import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Define structured array dtypes for nodes and edges
NODE_DTYPE = np.dtype([
    ("t", "i4"),     # Tick number
    ("gid", "i4"),   # Global ID
    ("type", "u1"),  # Node type: 0=epsilon, 1=theta
    ("x", "f4"),     # X position in km
    ("y", "f4"),     # Y position in km
    ("feat", "f4", (4,))  # Example features (placeholder for sensor data)
])

EDGE_DTYPE = np.dtype([
    ("t", "i4"),     # Tick number
    ("src", "i4"),   # Source node ID
    ("dst", "i4"),   # Destination node ID
    ("rel", "u1")    # Relationship type: 0=comm link, 1=detection
])


class SnapshotWriter:
    """
    Writer for serializing Aquascan simulation snapshots to HDF5 files.
    
    Creates an optimized, tick-indexed file format that's efficient for both:
    1. Random-access time slicing during model training
    2. Sequential streaming for visualization and analysis
    
    All node and edge data is synchronized by tick, allowing for coherent
    graph reconstruction at any point in the simulation timeline.
    """
    
    def __init__(self, out_path: Path, meta: Dict[str, Any], est_ticks: int = 600):
        """
        Initialize a new snapshot writer.
        
        Args:
            out_path: Path where the HDF5 file will be written
            meta: Dictionary of metadata to store in /globals (e.g., config)
            est_ticks: Estimated tick count for pre-allocation (optimization)
        """
        # Ensure directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 file
        self.f = h5py.File(out_path, "w")
        
        # Create resizable datasets with chunking and compression
        # Estimate ~600 nodes per tick (570 epsilon + some theta)
        est_nodes = est_ticks * 600
        self.nodes = self.f.create_dataset(
            "nodes", 
            shape=(0,), 
            maxshape=(None,), 
            dtype=NODE_DTYPE,
            chunks=(min(10_000, est_ticks*10),),  # Chunk by ~10 ticks
            compression="gzip", 
            compression_opts=4
        )
        
        # Estimate ~1200 edges per tick (1127 permanent + some detections)
        est_edges = est_ticks * 1200
        self.edges = self.f.create_dataset(
            "edges", 
            shape=(0,), 
            maxshape=(None,), 
            dtype=EDGE_DTYPE,
            chunks=(min(10_000, est_ticks*10),),  # Chunk by ~10 ticks
            compression="gzip", 
            compression_opts=4
        )
        
        # Store metadata as JSON
        self.f.create_dataset("globals", data=json.dumps(meta))
        
        # Track current sizes for append operations
        self._n_nodes = 0
        self._n_edges = 0
    
    def append(self, t: int, node_rows: np.ndarray, edge_rows: np.ndarray):
        """
        Append nodes and edges for a specific tick to the HDF5 file.
        
        Args:
            t: Current tick number
            node_rows: Array of node data with NODE_DTYPE structure
            edge_rows: Array of edge data with EDGE_DTYPE structure
        """
        n, e = len(node_rows), len(edge_rows)
        
        # Append nodes if any
        if n:
            self.nodes.resize(self._n_nodes + n, axis=0)
            self.nodes[self._n_nodes:self._n_nodes+n] = node_rows
            self._n_nodes += n
        
        # Append edges if any
        if e:
            self.edges.resize(self._n_edges + e, axis=0)
            self.edges[self._n_edges:self._n_edges+e] = edge_rows
            self._n_edges += e
    
    def close(self):
        """Close the HDF5 file, ensuring all data is properly written."""
        self.f.close()
        
    def __enter__(self):
        """Support context manager protocol."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure file is closed when context ends."""
        self.close()
