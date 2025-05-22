#!/usr/bin/env python3
"""
Data Inspection Utility for Processed Graph Datasets

This script inspects the structure and content of processed graph datasets created
by the dataset.build_graphs module. Use this for debugging data pipeline issues,
understanding graph structure, or verifying data integrity after processing.

When to use:
- After running dataset.build_graphs to verify output format
- When debugging GNN training issues related to data structure
- To understand node/edge features and label distributions
- To check class imbalance ratios before training

Usage:
    python inspect_data.py

Prerequisites:
    - data/processed/train.pt must exist (run dataset.build_graphs first)
"""

import torch

def inspect_data():
    """
    Inspect processed graph dataset structure and content.
    
    Provides detailed information about:
    - Graph count and structure
    - Node types and feature dimensions
    - Edge types and connectivity patterns
    - Label distribution for class imbalance analysis
    """
    try:
        # Load a small subset of training data
        train_graphs = torch.load("data/processed/train.pt", weights_only=False)
    except FileNotFoundError:
        print("Error: data/processed/train.pt not found.")
        print("Run 'python -m aquascan.dataset.build_graphs' first to process raw data.")
        return
    
    print(f"Dataset Overview:")
    print(f"Number of training graphs: {len(train_graphs)}")
    
    # Analyze first graph structure
    g = train_graphs[0]
    print(f"\nGraph Structure (first graph):")
    print(f"Node types: {g.node_types}")
    print(f"Edge types: {g.edge_types}")
    
    # Analyze node features
    if 'epsilon' in g.node_types:
        print(f"\nEpsilon (sensor) nodes:")
        print(f"  Count: {g['epsilon'].x.shape[0]}")
        print(f"  Feature dimension: {g['epsilon'].x.shape[1]}")
        print(f"  Sample features [x, y, Δx, Δy]: {g['epsilon'].x[:3]}")
    
    if 'theta' in g.node_types:
        print(f"\nTheta (marine entity) nodes:")
        print(f"  Count: {g['theta'].x.shape[0]}")
        print(f"  Feature dimension: {g['theta'].x.shape[1]}")
        print(f"  Sample features [x, y, Δx, Δy]: {g['theta'].x[:3]}")
    
    # Analyze edge structure and class imbalance
    total_pos_labels = 0
    total_predictions = 0
    
    for edge_type in g.edge_types:
        print(f"\nEdge type: {edge_type}")
        edge_data = g[edge_type]
        
        if hasattr(edge_data, 'edge_index'):
            print(f"  Edges: {edge_data.edge_index.shape[1]}")
        
        if hasattr(edge_data, 'edge_label_index'):
            print(f"  Prediction candidates: {edge_data.edge_label_index.shape[1]}")
            total_predictions += edge_data.edge_label_index.shape[1]
            
        if hasattr(edge_data, 'edge_label'):
            pos_count = edge_data.edge_label.sum().item()
            total_count = edge_data.edge_label.shape[0]
            print(f"  Positive labels: {pos_count} / {total_count}")
            print(f"  Class ratio: 1:{total_count/pos_count:.0f}" if pos_count > 0 else "  Class ratio: No positives")
            total_pos_labels += pos_count
    
    # Overall class imbalance summary
    if total_predictions > 0 and total_pos_labels > 0:
        print(f"\nClass Imbalance Summary:")
        print(f"Total prediction candidates across all graphs: ~{total_predictions * len(train_graphs):,}")
        print(f"Expected positive labels: ~{total_pos_labels * len(train_graphs)}")
        print(f"Overall ratio: 1:{(total_predictions * len(train_graphs)) / (total_pos_labels * len(train_graphs)):.0f}")
        print(f"This extreme imbalance requires class-balanced loss and threshold tuning.")

if __name__ == "__main__":
    inspect_data()