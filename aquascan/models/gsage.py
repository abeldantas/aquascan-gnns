#!/usr/bin/env python3
"""
Heterogeneous GraphSAGE Model for Marine Entity Detection Prediction

This module implements a 3-layer HeteroGraphSAGE model for link prediction
between epsilon nodes (sensors) and theta contacts (marine entities).

Model Architecture:
- Node types: epsilon (ε), theta (θ)  
- Edge types: epsilon-epsilon ('communicates'), epsilon-theta ('detects')
- Node features: [x, y, Δx, Δy] (position and velocity)
- 3-layer HeteroGraphSAGE with ReLU activation and batch normalization
- Hidden dimension: 64
- Readout: element-wise dot product of final embeddings + sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE model for marine entity detection prediction.
    
    The model processes heterogeneous graphs with epsilon (sensor) and theta (marine entity)
    nodes, predicting future detection links between them.
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 3):
        """
        Initialize the HeteroGraphSAGE model.
        
        Args:
            input_dim: Dimension of input node features (default: 4 for [x, y, Δx, Δy])
            hidden_dim: Hidden dimension for embeddings (default: 64)
            num_layers: Number of GraphSAGE layers (default: 3)
        """
        super(HeteroGraphSAGE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers        
        # Input projection for node features
        self.node_projections = nn.ModuleDict({
            'epsilon': nn.Linear(input_dim, hidden_dim),
            'theta': nn.Linear(input_dim, hidden_dim),
        })
        
        # Heterogeneous GraphSAGE layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Create HeteroConv layer with SAGEConv for each edge type
            conv_dict = {
                ('epsilon', 'communicates', 'epsilon'): SAGEConv(hidden_dim, hidden_dim),
                ('epsilon', 'detects', 'theta'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('theta', 'rev_detects', 'epsilon'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }
            
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.conv_layers.append(hetero_conv)
            
            # Batch normalization for each node type
            batch_norm_dict = nn.ModuleDict({
                'epsilon': nn.BatchNorm1d(hidden_dim),
                'theta': nn.BatchNorm1d(hidden_dim),
            })
            self.batch_norms.append(batch_norm_dict)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the HeteroGraphSAGE model.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Dictionary of final node embeddings for each node type
        """
        # Project input features to hidden dimension
        x_dict = {
            node_type: self.node_projections[node_type](x) 
            for node_type, x in x_dict.items()
        }
        
        # Pass through GraphSAGE layers
        for i, (conv, batch_norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # Apply convolution
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply batch normalization and ReLU (except for last layer)
            for node_type in x_dict:
                if x_dict[node_type].size(0) > 1:  # Only apply BatchNorm if batch size > 1
                    x_dict[node_type] = batch_norm[node_type](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])
        
        return x_dict    
    def decode(self, z_dict, edge_label_index):
        """
        Decode link predictions using element-wise dot product.
        
        Args:
            z_dict: Dictionary of node embeddings from forward pass
            edge_label_index: Edge indices for prediction [2, num_edges]
            
        Returns:
            Tensor of prediction scores (before sigmoid)
        """
        epsilon_emb = z_dict['epsilon']
        theta_emb = z_dict['theta']
        
        # Get embeddings for source (epsilon) and target (theta) nodes
        epsilon_indices = edge_label_index[0]
        theta_indices = edge_label_index[1]
        
        src_emb = epsilon_emb[epsilon_indices]  # [num_edges, hidden_dim]
        dst_emb = theta_emb[theta_indices]     # [num_edges, hidden_dim]
        
        # Element-wise dot product and sum across dimensions
        scores = (src_emb * dst_emb).sum(dim=1)  # [num_edges]
        
        return scores
    
    def predict(self, x_dict, edge_index_dict, edge_label_index):
        """
        Make predictions for given node features and edge indices.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            edge_label_index: Edge indices for prediction
            
        Returns:
            Tensor of prediction probabilities (after sigmoid)
        """
        # Get node embeddings
        z_dict = self.forward(x_dict, edge_index_dict)
        
        # Decode link predictions
        scores = self.decode(z_dict, edge_label_index)
        
        # Apply sigmoid for probabilities
        probabilities = torch.sigmoid(scores)
        
        return probabilities


def create_model(input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 3) -> HeteroGraphSAGE:
    """
    Factory function to create a HeteroGraphSAGE model with standard parameters.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden embedding dimension  
        num_layers: Number of GraphSAGE layers
        
    Returns:
        Initialized HeteroGraphSAGE model
    """
    return HeteroGraphSAGE(input_dim, hidden_dim, num_layers)