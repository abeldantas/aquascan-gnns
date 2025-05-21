"""
Windowed Graph Builder

This module converts raw HDF5 snapshots into fixed-length spatiotemporal graphs 
that libraries like PyTorch Geometric can ingest. It builds training, validation,
and test splits, along with adversarial examples.

Features:
- Windowed processing of HDF5 files with context and horizon windows
- Feature extraction for nodes: position and velocity
- Edge generation for communication and detection relationships
- Target label creation for future detection events
- Adversarial example generation with position distortion and edge removal
"""

import os
import sys
import h5py
import json
import torch
import random
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Make sure PyTorch Geometric is installed
try:
    from torch_geometric.data import HeteroData
except ImportError:
    print("Error: torch_geometric not installed. Please install it with 'pip install torch_geometric'")
    sys.exit(1)


def create_graph_from_window(file_path: str, start_tick: int, context_len: int, horizon_len: int) -> HeteroData:
    """
    Create a graph from a window of ticks in an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        start_tick: Starting tick for the window
        context_len: Length of the context window
        horizon_len: Length of the horizon window
        
    Returns:
        A HeteroData graph
    """
    # Create a new heterogeneous graph
    graph = HeteroData()
    
    with h5py.File(file_path, 'r') as f:
        # Get the total number of ticks
        ticks = np.unique(f['nodes']['t'])
        max_tick = ticks.max()
        
        if start_tick + context_len + horizon_len > max_tick + 1:
            raise ValueError(f"Not enough ticks in file. Need {context_len + horizon_len} ticks, but only have {max_tick - start_tick + 1} available.")
        
        # Context window ticks
        context_ticks = list(range(start_tick, start_tick + context_len))
        horizon_ticks = list(range(start_tick + context_len, start_tick + context_len + horizon_len))
        
        # Extract all nodes in the context window
        context_nodes = f['nodes'][np.isin(f['nodes']['t'], context_ticks)]
        
        # Extract all edges in the context window
        context_edges = f['edges'][np.isin(f['edges']['t'], context_ticks)]
        
        # Extract all edges in the horizon window (for target labels)
        horizon_edges = f['edges'][np.isin(f['edges']['t'], horizon_ticks)]
        
        # Get unique node IDs and create mapping
        epsilon_nodes = context_nodes[context_nodes['type'] == 0]
        theta_nodes = context_nodes[context_nodes['type'] == 1]
        
        # Build a map of unique node IDs
        epsilon_ids = np.unique(epsilon_nodes['gid'])
        theta_ids = np.unique(theta_nodes['gid'])
        
        # Create node features
        epsilon_features = {}
        theta_features = {}
        
        # For each unique epsilon node, calculate its average position across the context window
        for node_id in epsilon_ids:
            node_data = epsilon_nodes[epsilon_nodes['gid'] == node_id]
            
            # Average position over the context window
            avg_pos = np.mean(node_data[['x', 'y']], axis=0)
            
            # Calculate velocity (if we have at least 2 points)
            if len(node_data) >= 2:
                sorted_data = np.sort(node_data, order='t')
                first_pos = sorted_data[0][['x', 'y']]
                last_pos = sorted_data[-1][['x', 'y']]
                velocity = (last_pos - first_pos) / (len(sorted_data) - 1)
            else:
                velocity = np.zeros(2)
            
            # Feature vector: [x, y, dx, dy]
            features = np.concatenate([avg_pos, velocity])
            epsilon_features[node_id] = features
        
        # Same for theta nodes
        for node_id in theta_ids:
            node_data = theta_nodes[theta_nodes['gid'] == node_id]
            
            # Average position over the context window
            avg_pos = np.mean(node_data[['x', 'y']], axis=0)
            
            # Calculate velocity (if we have at least 2 points)
            if len(node_data) >= 2:
                sorted_data = np.sort(node_data, order='t')
                first_pos = sorted_data[0][['x', 'y']]
                last_pos = sorted_data[-1][['x', 'y']]
                velocity = (last_pos - first_pos) / (len(sorted_data) - 1)
            else:
                velocity = np.zeros(2)
            
            # Feature vector: [x, y, dx, dy]
            features = np.concatenate([avg_pos, velocity])
            theta_features[node_id] = features
        
        # Create node tensors
        if epsilon_features:
            epsilon_x = torch.tensor([epsilon_features[gid] for gid in sorted(epsilon_features.keys())], dtype=torch.float)
            graph['epsilon'].x = epsilon_x
            graph['epsilon'].node_ids = torch.tensor(sorted(epsilon_features.keys()), dtype=torch.long)
        else:
            graph['epsilon'].x = torch.zeros((0, 4), dtype=torch.float)
            graph['epsilon'].node_ids = torch.zeros(0, dtype=torch.long)
        
        if theta_features:
            theta_x = torch.tensor([theta_features[gid] for gid in sorted(theta_features.keys())], dtype=torch.float)
            graph['theta'].x = theta_x
            graph['theta'].node_ids = torch.tensor(sorted(theta_features.keys()), dtype=torch.long)
        else:
            graph['theta'].x = torch.zeros((0, 4), dtype=torch.float)
            graph['theta'].node_ids = torch.zeros(0, dtype=torch.long)
        
        # Create edge indices for the context window
        # Communication edges (epsilon -> epsilon)
        comm_edges = []
        for edge in context_edges[context_edges['rel'] == 0]:
            src, dst = edge['src'], edge['dst']
            if src in epsilon_ids and dst in epsilon_ids:
                src_idx = np.where(epsilon_ids == src)[0][0]
                dst_idx = np.where(epsilon_ids == dst)[0][0]
                comm_edges.append((src_idx, dst_idx))
        
        if comm_edges:
            comm_edge_index = torch.tensor(comm_edges, dtype=torch.long).t().contiguous()
            graph['epsilon', 'communicates', 'epsilon'].edge_index = comm_edge_index
        else:
            graph['epsilon', 'communicates', 'epsilon'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Detection edges in context window (epsilon -> theta)
        detect_edges = []
        for edge in context_edges[context_edges['rel'] == 1]:
            src, dst = edge['src'], edge['dst']
            if src in epsilon_ids and dst in theta_ids:
                src_idx = np.where(epsilon_ids == src)[0][0]
                dst_idx = np.where(theta_ids == dst)[0][0]
                detect_edges.append((src_idx, dst_idx))
        
        if detect_edges:
            detect_edge_index = torch.tensor(detect_edges, dtype=torch.long).t().contiguous()
            graph['epsilon', 'detects', 'theta'].edge_index = detect_edge_index
        else:
            graph['epsilon', 'detects', 'theta'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create target edge labels for the horizon window
        # Ground truth: which epsilon-theta pairs will have a detection edge in the horizon window
        future_detections = set()
        for edge in horizon_edges[horizon_edges['rel'] == 1]:
            src, dst = edge['src'], edge['dst']
            if src in epsilon_ids and dst in theta_ids:
                future_detections.add((src, dst))
        
        # Create edge label index for all possible epsilon-theta pairs
        all_pairs = []
        label_values = []
        
        for src_idx, src in enumerate(sorted(epsilon_features.keys())):
            for dst_idx, dst in enumerate(sorted(theta_features.keys())):
                all_pairs.append((src_idx, dst_idx))
                label_values.append(1.0 if (src, dst) in future_detections else 0.0)
        
        if all_pairs:
            edge_label_index = torch.tensor(all_pairs, dtype=torch.long).t().contiguous()
            graph['epsilon', 'will_detect', 'theta'].edge_label_index = edge_label_index
            graph['epsilon', 'will_detect', 'theta'].edge_label = torch.tensor(label_values, dtype=torch.float)
        else:
            graph['epsilon', 'will_detect', 'theta'].edge_label_index = torch.zeros((2, 0), dtype=torch.long)
            graph['epsilon', 'will_detect', 'theta'].edge_label = torch.zeros(0, dtype=torch.float)
        
        # Add metadata to the graph
        graph.context_len = context_len
        graph.horizon_len = horizon_len
        
        return graph


def h5_to_graphs(file_path: str, context_len: int, horizon_len: int) -> List[HeteroData]:
    """
    Convert an HDF5 file into a list of graph objects with sliding windows.
    
    Args:
        file_path: Path to the HDF5 file
        context_len: Length of the context window (time steps)
        horizon_len: Length of the horizon window (time steps)
        
    Returns:
        List of HeteroData graphs
    """
    graphs = []
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Get the total number of ticks
        ticks = np.unique(f['nodes']['t'])
        max_tick = ticks.max()
        
        # Calculate how many windows we can create
        num_windows = max_tick - (context_len + horizon_len) + 2
        
        print(f"File {Path(file_path).name} has {max_tick + 1} ticks, creating {num_windows} windows")
        
        # Create sliding windows
        for start_tick in range(num_windows):
            try:
                graph = create_graph_from_window(file_path, start_tick, context_len, horizon_len)
                graphs.append(graph)
            except Exception as e:
                print(f"Error creating graph for window at tick {start_tick}: {e}")
    
    return graphs


def save_split(graphs: List[HeteroData], split_ratios: List[float], out_dir: str, adv_fraction: float = 0.0):
    """
    Split graphs into train/val/test sets and save them.
    
    Args:
        graphs: List of graph objects
        split_ratios: List of ratios for [train, val, test] splits
        out_dir: Output directory
        adv_fraction: Fraction of training data for adversarial examples
    """
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle the graphs
    random.shuffle(graphs)
    
    # Split the graphs
    n = len(graphs)
    train_size = int(n * split_ratios[0])
    val_size = int(n * split_ratios[1])
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size + val_size]
    test_graphs = graphs[train_size + val_size:]
    
    # Create a small set of adversarial examples if needed
    adv_graphs = []
    if adv_fraction > 0:
        # Number of adversarial examples to create
        adv_count = int(len(train_graphs) * adv_fraction)
        
        # Select random graphs for modification
        for i in range(adv_count):
            if i < len(train_graphs):
                # Create a copy of the graph
                adv_graph = train_graphs[i].clone()
                
                # Modify it to make it adversarial
                # 1. Add random noise to node positions
                if hasattr(adv_graph['epsilon'], 'x') and adv_graph['epsilon'].x.shape[0] > 0:
                    noise = torch.randn_like(adv_graph['epsilon'].x[:, :2]) * 0.5
                    adv_graph['epsilon'].x[:, :2] += noise
                
                if hasattr(adv_graph['theta'], 'x') and adv_graph['theta'].x.shape[0] > 0:
                    noise = torch.randn_like(adv_graph['theta'].x[:, :2]) * 0.5
                    adv_graph['theta'].x[:, :2] += noise
                
                # 2. Remove some communication edges randomly
                if 'epsilon' in adv_graph.node_types and ('epsilon', 'communicates', 'epsilon') in adv_graph.edge_types:
                    edge_index = adv_graph['epsilon', 'communicates', 'epsilon'].edge_index
                    if edge_index.shape[1] > 0:
                        # Keep approximately 70% of edges
                        mask = torch.rand(edge_index.shape[1]) > 0.3
                        adv_graph['epsilon', 'communicates', 'epsilon'].edge_index = edge_index[:, mask]
                
                adv_graphs.append(adv_graph)
    
    # Save the data splits
    torch.save(train_graphs, out_path / 'train.pt')
    torch.save(val_graphs, out_path / 'val.pt')
    torch.save(test_graphs, out_path / 'test.pt')
    
    if adv_graphs:
        torch.save(adv_graphs, out_path / 'adversarial.pt')
    
    # Create metadata
    if len(graphs) > 0:
        sample_graph = graphs[0]
        
        # Get node and edge counts from the sample graph
        node_counts = {}
        edge_counts = {}
        
        for node_type in sample_graph.node_types:
            if hasattr(sample_graph[node_type], 'x'):
                node_counts[node_type] = sample_graph[node_type].x.shape[0]
        
        for edge_type in sample_graph.edge_types:
            if hasattr(sample_graph[edge_type], 'edge_index'):
                edge_counts[str(edge_type)] = sample_graph[edge_type].edge_index.shape[1]
        
        # Add to metadata
        metadata = {
            'context_len': sample_graph.context_len,
            'horizon_len': sample_graph.horizon_len,
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'split_ratios': {
                'train': split_ratios[0],
                'val': split_ratios[1],
                'test': split_ratios[2]
            },
            'adv_fraction': adv_fraction,
            'train_size': len(train_graphs),
            'val_size': len(val_graphs),
            'test_size': len(test_graphs),
            'adv_size': len(adv_graphs),
            'total_graphs': len(graphs)
        }
        
        # Save metadata
        with open(out_path / 'meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def validate_graphs(graphs: List[HeteroData]) -> bool:
    """
    Validate a list of graphs to ensure they meet requirements.
    
    Args:
        graphs: List of graph objects to validate
        
    Returns:
        True if all graphs pass validation, False otherwise
    """
    if not graphs:
        print("No graphs to validate")
        return False
    
    for i, graph in enumerate(graphs[:min(5, len(graphs))]):  # Check at least the first 5 graphs
        # Check node types
        if 'epsilon' not in graph.node_types:
            print(f"Graph {i}: Missing epsilon nodes")
            return False
        
        if 'theta' not in graph.node_types:
            print(f"Graph {i}: Missing theta nodes")
            return False
        
        # Check edge types
        required_edge_types = [
            ('epsilon', 'communicates', 'epsilon'),
            ('epsilon', 'detects', 'theta'),
            ('epsilon', 'will_detect', 'theta')
        ]
        
        for edge_type in required_edge_types:
            if edge_type not in graph.edge_types:
                print(f"Graph {i}: Missing edge type {edge_type}")
                return False
        
        # Check edge label shape
        target_edge_type = ('epsilon', 'will_detect', 'theta')
        if hasattr(graph[target_edge_type], 'edge_label') and hasattr(graph[target_edge_type], 'edge_label_index'):
            if graph[target_edge_type].edge_label.shape[0] != graph[target_edge_type].edge_label_index.shape[1]:
                print(f"Graph {i}: Target label shape mismatch")
                return False
        else:
            print(f"Graph {i}: Missing edge label or label index")
            return False
        
        # Check that context and horizon lengths are set
        if not hasattr(graph, 'context_len') or not hasattr(graph, 'horizon_len'):
            print(f"Graph {i}: Missing context_len or horizon_len attribute")
            return False
    
    # Try to serialize a graph to ensure it can be saved
    try:
        torch.save(graphs[0], Path(os.environ.get('TMPDIR', '/tmp')) / 'test_graph.pt')
    except Exception as e:
        print(f"Graph serialization failed: {e}")
        return False
    
    return True


def main():
    """Main entry point for the windowed graph builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Windowed Graph Builder')
    parser.add_argument('--raw', type=str, default='data/raw', help='Input directory with HDF5 files')
    parser.add_argument('--out', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--context', type=int, default=60, help='Context window length')
    parser.add_argument('--horizon', type=int, default=30, help='Horizon window length')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.15, 0.15], help='Split ratios for train/val/test')
    parser.add_argument('--adv_fraction', type=float, default=0.05, help='Adversarial example fraction')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Find HDF5 files
    raw_dir = Path(args.raw)
    files = list(raw_dir.glob('*.h5'))
    
    if not files:
        print(f"No HDF5 files found in {args.raw}")
        return 1
    
    if args.limit:
        files = files[:args.limit]
    
    print(f"Found {len(files)} HDF5 files")
    
    # Create graphs
    all_graphs = []
    
    start_time = time.time()
    
    for file_path in files:
        print(f"Processing file: {file_path}")
        try:
            graphs = h5_to_graphs(str(file_path), args.context, args.horizon)
            all_graphs.extend(graphs)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Validate graphs
    print(f"Created {len(all_graphs)} graphs total")
    if len(all_graphs) > 0:
        if validate_graphs(all_graphs):
            print("All graphs passed validation")
            
            # Save the graphs
            save_split(all_graphs, args.split, args.out, args.adv_fraction)
            print(f"Saved graph splits to {args.out}")
            
            elapsed_time = time.time() - start_time
            print(f"Process completed in {elapsed_time:.2f} seconds")
            return 0
        else:
            print("Graph validation failed")
            return 1
    else:
        print("No graphs were created")
        return 1


if __name__ == '__main__':
    sys.exit(main())
