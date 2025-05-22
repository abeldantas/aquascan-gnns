#!/usr/bin/env python3
"""
Kalman Filter Baseline for Marine Entity Detection Prediction

This script implements a classical constant-velocity Kalman filter benchmark
that predicts future ε-θ detection links on the processed test split.

Implementation follows the specification:
1. Per-θ Kalman filter with state vector [x, y, vx, vy]ᵀ
2. Use last context_len positions to initialize/update
3. Predict horizon_len steps ahead
4. Reconstruct edges based on 100m distance threshold
5. Calculate AUC, precision, recall metrics
"""

import json
import pathlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from typing import Dict, List, Tuple


def run_kf(obs_xy: np.ndarray, horizon: int) -> np.ndarray:
    """
    Run a constant-velocity Kalman filter for trajectory prediction.
    
    Args:
        obs_xy: T×2 array (context positions)
        horizon: Number of steps to predict ahead
        
    Returns:
        horizon×2 array of predicted positions
    """
    if len(obs_xy) < 1:
        return np.zeros((horizon, 2))
    
    if len(obs_xy) == 1:
        # Only one observation, predict same position
        return np.tile(obs_xy[-1], (horizon, 1))
    
    # Initialize state [x, y, vx, vy]
    dt = 1.0  # Time step in seconds
    last_pos = obs_xy[-1]
    
    # Estimate initial velocity from last two positions
    if len(obs_xy) >= 2:
        velocity = obs_xy[-1] - obs_xy[-2]
    else:
        velocity = np.zeros(2)
    
    # State vector [x, y, vx, vy]
    x = np.array([last_pos[0], last_pos[1], velocity[0], velocity[1]])
    
    # State transition matrix (constant velocity)
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Process noise covariance
    q = 0.01  # Process noise
    Q = np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ]) * q**2
    
    # Initial covariance
    P = np.eye(4) * 0.1
    
    # If multiple observations, update the filter
    if len(obs_xy) > 2:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observe position only
        R = np.eye(2) * 0.01  # Measurement noise
        
        # Update with observations (skip first two used for initialization)
        for i in range(2, len(obs_xy)):
            # Predict
            x = A @ x
            P = A @ P @ A.T + Q
            
            # Update with measurement
            z = obs_xy[i]
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
    
    # Predict horizon steps
    predictions = np.zeros((horizon, 2))
    state = x.copy()
    
    for i in range(horizon):
        state = A @ state
        predictions[i] = state[:2]
    
    return predictions


def evaluate_file(file_pt: str) -> Dict[str, float]:
    """
    Evaluate Kalman filter on dataset file.
    
    Args:
        file_pt: Path to .pt file containing graphs
        
    Returns:
        Dictionary with AUC, Precision, Recall
    """
    graphs = torch.load(file_pt, weights_only=False)
    y_true, y_pred = [], []
    
    print(f"Processing {len(graphs)} graphs...")
    
    for i, g in enumerate(graphs):
        if i % 10 == 0:
            print(f"Processing graph {i+1}/{len(graphs)}")
        
        context_len = int(g.context_len)
        horizon_len = int(g.horizon_len)
        
        # Get epsilon and theta features [x, y, vx, vy]
        epsilon_features = g['epsilon'].x.numpy()
        theta_features = g['theta'].x.numpy()
        
        # Build observation dict {theta_gid: [xy_t0 ... xy_t{ctx-1}]}
        theta_obs = {}
        
        # Since we only have current state, reconstruct context using velocities
        for theta_idx in range(len(theta_features)):
            current_pos = theta_features[theta_idx, :2]
            current_vel = theta_features[theta_idx, 2:]
            
            # Reconstruct context by going backward in time
            obs_positions = []
            for t in range(context_len):
                # Position at time (current - t)
                past_pos = current_pos - current_vel * t
                obs_positions.append(past_pos)
            
            # Reverse to get chronological order
            obs_positions.reverse()
            theta_obs[theta_idx] = np.array(obs_positions)
        
        # Predict horizon steps for each theta
        theta_predictions = {}
        for theta_idx, obs_xy in theta_obs.items():
            pred = run_kf(obs_xy, horizon_len)
            theta_predictions[theta_idx] = pred
        
        # Get epsilon positions
        epsilon_positions = epsilon_features[:, :2]
        
        # Get labels and indices for prediction task
        edge_labels = g[('epsilon', 'will_detect', 'theta')].edge_label.numpy()
        edge_label_index = g[('epsilon', 'will_detect', 'theta')].edge_label_index.numpy()
        
        # Distance threshold: 100m = 0.1km
        distance_threshold = 0.1
        
        # Generate predictions
        predictions = []
        for edge_idx in range(len(edge_labels)):
            epsilon_idx = edge_label_index[0, edge_idx]
            theta_idx = edge_label_index[1, edge_idx]
            
            epsilon_pos = epsilon_positions[epsilon_idx]
            
            if theta_idx in theta_predictions:
                # Use first prediction (horizon_len=1)
                predicted_theta_pos = theta_predictions[theta_idx][0]
                distance = np.linalg.norm(epsilon_pos - predicted_theta_pos)
                
                # Binary prediction based on distance threshold
                prediction = 1.0 if distance <= distance_threshold else 0.0
            else:
                prediction = 0.0
            
            predictions.append(prediction)
        
        y_true.extend(edge_labels)
        y_pred.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print(f"Total predictions: {len(y_true)}")
    print(f"Positive labels: {y_true.sum()}")
    print(f"Predicted positive: {y_pred.sum()}")
    
    # Calculate metrics
    if len(np.unique(y_pred)) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = 0.5
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC": float(auc),
        "Precision": float(precision),
        "Recall": float(recall)
    }


if __name__ == "__main__":
    print("Running Kalman Filter Baseline Evaluation")
    
    res = evaluate_file("data/processed/test.pt")
    
    pathlib.Path("results").mkdir(exist_ok=True)
    
    with open("results/kalman_baseline.json", "w") as f:
        json.dump(res, f, indent=2)
    
    print("Results:")
    for key, value in res.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nResults saved to results/kalman_baseline.json")
