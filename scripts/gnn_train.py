#!/usr/bin/env python3
"""
GraphSAGE Training Script

This script implements the training loop for the HeteroGraphSAGE model
on the processed marine entity detection dataset.

Training Configuration:
- Model: 3-layer HeteroGraphSAGE with 64 hidden dim
- Loss: BCEWithLogitsLoss on edge labels
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Mini-batch: NeighborLoader with fan-out [10,10,10]
- Early stopping: 5 epochs without validation AUC improvement
- Target: Val AUC ≥ 0.83
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

from aquascan.models.gsage import create_model

def load_data(data_dir: str) -> Tuple[List, List]:
    """
    Load training and validation datasets.
    
    Args:
        data_dir: Directory containing train.pt and val.pt
        
    Returns:
        Tuple of (train_graphs, val_graphs)
    """
    train_path = Path(data_dir) / "train.pt"
    val_path = Path(data_dir) / "val.pt"
    
    print(f"Loading training data from {train_path}")
    train_graphs = torch.load(train_path, weights_only=False)
    
    print(f"Loading validation data from {val_path}")
    val_graphs = torch.load(val_path, weights_only=False)
    
    print(f"Loaded {len(train_graphs)} training graphs, {len(val_graphs)} validation graphs")
    
    return train_graphs, val_graphs


def create_neighbor_loader(graphs, batch_size: int = 8, num_neighbors: List[int] = [10, 10, 10]):
    """
    Create a NeighborLoader for mini-batch training.
    
    Args:
        graphs: List of graph objects
        batch_size: Number of graphs per batch
        num_neighbors: Fan-out for neighbor sampling
        
    Returns:
        DataLoader for mini-batch sampling
    """
    # For now, we'll use simple batching since NeighborLoader requires
    # specific graph structure that may not match our data format
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return batch
    
    return DataLoader(graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def evaluate_model(model, graphs, device) -> Dict[str, float]:
    """
    Evaluate model on a set of graphs.
    
    Args:
        model: Trained HeteroGraphSAGE model
        graphs: List of graphs to evaluate on  
        device: Device to run evaluation on
        
    Returns:
        Dictionary with AUC, Precision, Recall metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            
            # Prepare node features
            x_dict = {
                'epsilon': graph['epsilon'].x,
                'theta': graph['theta'].x,
            }
            
            # Prepare edge indices
            edge_index_dict = {}
            if ('epsilon', 'communicates', 'epsilon') in graph.edge_types:
                edge_index_dict[('epsilon', 'communicates', 'epsilon')] = graph[('epsilon', 'communicates', 'epsilon')].edge_index
            if ('epsilon', 'detects', 'theta') in graph.edge_types:
                edge_index_dict[('epsilon', 'detects', 'theta')] = graph[('epsilon', 'detects', 'theta')].edge_index
                # Add reverse edge
                edge_index_dict[('theta', 'rev_detects', 'epsilon')] = graph[('epsilon', 'detects', 'theta')].edge_index.flip(0)
            
            # Get predictions
            edge_label_index = graph[('epsilon', 'will_detect', 'theta')].edge_label_index
            edge_labels = graph[('epsilon', 'will_detect', 'theta')].edge_label
            
            predictions = model.predict(x_dict, edge_index_dict, edge_label_index)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(edge_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    
    # Convert probabilities to binary predictions for precision/recall
    binary_preds = (all_preds > 0.5).astype(int)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    
    return {
        "AUC": float(auc),
        "Precision": float(precision), 
        "Recall": float(recall)
    }

def calculate_class_weights(graphs):
    """Calculate class weights for imbalanced dataset."""
    pos_count = 0
    neg_count = 0
    
    for graph in graphs:
        labels = graph[('epsilon', 'will_detect', 'theta')].edge_label
        pos_count += labels.sum().item()
        neg_count += (1 - labels).sum().item()
    
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Class distribution: {pos_count} positive, {neg_count} negative")
    print(f"Positive class weight: {pos_weight:.2f}")
    
    return pos_weight

def train_epoch(model, train_loader, optimizer, device, pos_weight=1.0) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: HeteroGraphSAGE model
        train_loader: DataLoader for training graphs
        optimizer: Optimizer
        device: Device to train on
        pos_weight: Weight for positive class in BCEWithLogitsLoss
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Create class-balanced loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    for batch in train_loader:
        optimizer.zero_grad()
        batch_loss = 0
        
        for graph in batch:
            graph = graph.to(device)
            
            # Prepare node features
            x_dict = {
                'epsilon': graph['epsilon'].x,
                'theta': graph['theta'].x,
            }
            
            # Prepare edge indices
            edge_index_dict = {}
            if ('epsilon', 'communicates', 'epsilon') in graph.edge_types:
                edge_index_dict[('epsilon', 'communicates', 'epsilon')] = graph[('epsilon', 'communicates', 'epsilon')].edge_index
            if ('epsilon', 'detects', 'theta') in graph.edge_types:
                edge_index_dict[('epsilon', 'detects', 'theta')] = graph[('epsilon', 'detects', 'theta')].edge_index
                # Add reverse edge
                edge_index_dict[('theta', 'rev_detects', 'epsilon')] = graph[('epsilon', 'detects', 'theta')].edge_index.flip(0)
            
            # Forward pass
            z_dict = model.forward(x_dict, edge_index_dict)
            
            # Get prediction targets
            edge_label_index = graph[('epsilon', 'will_detect', 'theta')].edge_label_index
            edge_labels = graph[('epsilon', 'will_detect', 'theta')].edge_label.float()
            
            # Decode predictions
            scores = model.decode(z_dict, edge_label_index)
            
            # Calculate class-balanced loss
            loss = criterion(scores, edge_labels)
            batch_loss += loss
        
        # Backward pass
        if batch_loss > 0:
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)

def train_model(train_graphs, val_graphs, epochs: int = 50, batch_size: int = 8, 
                lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 5,
                device = None) -> Dict:
    """
    Main training loop with early stopping.
    
    Args:
        train_graphs: Training graphs
        val_graphs: Validation graphs
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience
        device: Device to train on
        
    Returns:
        Dictionary with training history and best model info
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    
    # Create model
    model = create_model(input_dim=4, hidden_dim=64, num_layers=3)
    model = model.to(device)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate class weights for imbalanced dataset
    pos_weight = calculate_class_weights(train_graphs)
    
    # Create data loaders
    train_loader = create_neighbor_loader(train_graphs, batch_size=batch_size)
    
    # Training history
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': []
    }
    
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    
    print(f"Starting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train for one epoch with class-balanced loss
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_graphs, device)
        val_auc = val_metrics['AUC']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_metrics['Precision'])
        history['val_recall'].append(val_metrics['Recall'])
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Prec: {val_metrics['Precision']:.4f} | "
              f"Val Rec: {val_metrics['Recall']:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best.pt")
            
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
            break
    
    return {
        'history': history,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'final_epoch': epoch + 1
    }

def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description="Train HeteroGraphSAGE model")
    parser.add_argument("--data", default="data/processed", help="Data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_graphs, val_graphs = load_data(args.data)
    
    # Train model
    results = train_model(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device
    )
    
    # Save validation results
    Path("results").mkdir(exist_ok=True)
    val_results = {
        'AUC': results['best_val_auc'],
        'best_epoch': results['best_epoch'],
        'final_epoch': results['final_epoch']
    }
    
    with open("results/gnn_val.json", "w") as f:
        json.dump(val_results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation AUC: {results['best_val_auc']:.4f}")
    print(f"Results saved to results/gnn_val.json")
    print(f"Best model saved to checkpoints/best.pt")


if __name__ == "__main__":
    main()