#!/usr/bin/env python3
"""
GraphSAGE Evaluation Script

This script loads a trained HeteroGraphSAGE model checkpoint and evaluates
it on the test dataset, saving results in JSON format.

Usage:
    python scripts/gnn_eval.py --ckpt checkpoints/best.pt --data data/processed/test.pt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve

from aquascan.models.gsage import create_model


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.8) -> Tuple[float, float, float]:
    """
    Find optimal threshold for given minimum precision.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        min_precision: Minimum required precision
        
    Returns:
        Tuple of (optimal_threshold, precision_at_threshold, recall_at_threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find thresholds where precision >= min_precision
    valid_indices = np.where(precision >= min_precision)[0]
    
    if len(valid_indices) == 0:
        # No threshold achieves min_precision, use best available
        best_idx = np.argmax(precision)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        precision_at_threshold = precision[best_idx]
        recall_at_threshold = recall[best_idx]
        print(f"Warning: Could not achieve precision >= {min_precision}")
        print(f"Best achievable precision: {precision_at_threshold:.4f}")
    else:
        # Among valid thresholds, choose the one with highest recall
        best_valid_idx = valid_indices[np.argmax(recall[valid_indices])]
        optimal_threshold = thresholds[best_valid_idx] if best_valid_idx < len(thresholds) else 0.5
        precision_at_threshold = precision[best_valid_idx]
        recall_at_threshold = recall[best_valid_idx]
    
    return optimal_threshold, precision_at_threshold, recall_at_threshold


def load_model_and_evaluate(ckpt_path: str, test_data_path: str, device=None) -> Dict[str, float]:
    """
    Load trained model and evaluate on test data.
    
    Args:
        ckpt_path: Path to model checkpoint
        test_data_path: Path to test.pt file
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {ckpt_path}")
    print(f"Loading test data from {test_data_path}")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(input_dim=4, hidden_dim=64, num_layers=3)
    
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_graphs = torch.load(test_data_path, weights_only=False)
    print(f"Loaded {len(test_graphs)} test graphs")
    
    # Evaluate
    all_preds = []
    all_labels = []    
    with torch.no_grad():
        for i, graph in enumerate(test_graphs):
            if i % 10 == 0:
                print(f"Processing graph {i+1}/{len(test_graphs)}")
                
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
    
    print(f"Total predictions: {len(all_preds)}")
    print(f"Positive labels: {all_labels.sum()}")
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    
    # Find optimal threshold for precision >= 0.8
    optimal_threshold, precision_at_threshold, recall_at_threshold = find_optimal_threshold(
        all_labels, all_preds, min_precision=0.8
    )
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Predicted positive at optimal threshold: {(all_preds >= optimal_threshold).sum()}")
    
    # Also calculate metrics at default 0.5 threshold for comparison
    binary_preds_05 = (all_preds > 0.5).astype(int)
    precision_05 = precision_score(all_labels, binary_preds_05, zero_division=0)
    recall_05 = recall_score(all_labels, binary_preds_05, zero_division=0)
    
    return {
        "AUC": float(auc),
        "Precision_at_0.5": float(precision_05),
        "Recall_at_0.5": float(recall_05),
        "Optimal_Threshold": float(optimal_threshold),
        "Precision_at_Optimal": float(precision_at_threshold),
        "Recall_at_Optimal": float(recall_at_threshold)
    }

def main():
    """Main evaluation function with command line interface."""
    parser = argparse.ArgumentParser(description="Evaluate trained HeteroGraphSAGE model")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", default="data/processed/test.pt", help="Path to test data")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.ckpt).exists():
        print(f"Error: Checkpoint file {args.ckpt} not found")
        return
    
    if not Path(args.data).exists():
        print(f"Error: Test data file {args.data} not found")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate model
    results = load_model_and_evaluate(args.ckpt, args.data, device)
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    
    with open("results/gnn_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest Results:")
    print(f"  AUC: {results['AUC']:.4f}")
    print(f"  Precision @ 0.5: {results['Precision_at_0.5']:.4f}")
    print(f"  Recall @ 0.5: {results['Recall_at_0.5']:.4f}")
    print(f"  Optimal Threshold: {results['Optimal_Threshold']:.4f}")
    print(f"  Precision @ Optimal: {results['Precision_at_Optimal']:.4f}")
    print(f"  Recall @ Optimal: {results['Recall_at_Optimal']:.4f}")
    
    print(f"\nResults saved to results/gnn_test.json")
    
    # Check acceptance criteria
    if results["AUC"] >= 0.82:
        print(f"✅ Test AUC criterion met: {results['AUC']:.4f} ≥ 0.82")
    else:
        print(f"❌ Test AUC criterion not met: {results['AUC']:.4f} < 0.82")
        
    if results["Precision_at_Optimal"] >= 0.8:
        print(f"✅ Precision criterion met: {results['Precision_at_Optimal']:.4f} ≥ 0.8")
    else:
        print(f"❌ Precision criterion not met: {results['Precision_at_Optimal']:.4f} < 0.8")


if __name__ == "__main__":
    main()