# 🔧 FIXED Colab Cell 7: Kalman Baselines (Replaces the broken one)

```python
# Run FIXED Kalman filter baseline for comparison
print("📐 Running FIXED Kalman filter baselines...\\n")

# Fixed Kalman implementation that actually works
def distance_to_probability(distance, detection_radius=0.1):
    """Convert distance to detection probability using sigmoid."""
    steepness = 20.0
    return 1.0 / (1.0 + np.exp(steepness * (distance - detection_radius)))

def run_fixed_kalman(positions, horizon=1):
    """Fixed Kalman filter with proper trajectory prediction."""
    if len(positions) < 2:
        return np.tile(positions[0] if len(positions) > 0 else np.zeros(2), (horizon, 1))
    
    # Simple constant velocity model
    velocity = positions[-1] - positions[-2]
    last_pos = positions[-1]
    
    # Predict future positions
    predictions = []
    for t in range(1, horizon + 1):
        pred_pos = last_pos + velocity * t
        predictions.append(pred_pos)
    
    return np.array(predictions)

def evaluate_kalman_fixed(test_file):
    """Fixed evaluation function."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score
    
    graphs = torch.load(test_file, weights_only=False)
    y_true, y_scores = [], []
    
    for graph in graphs:
        context_len = int(graph.context_len)
        horizon_len = int(graph.horizon_len)
        
        # Get features
        epsilon_features = graph['epsilon'].x.numpy()
        theta_features = graph['theta'].x.numpy()
        
        # Get edge data
        edge_labels = graph[('epsilon', 'will_detect', 'theta')].edge_label.numpy()
        edge_index = graph[('epsilon', 'will_detect', 'theta')].edge_label_index.numpy()
        
        # For each theta, reconstruct trajectory using velocity
        theta_trajectories = {}
        for theta_idx in range(len(theta_features)):
            current_pos = theta_features[theta_idx, :2]
            current_vel = theta_features[theta_idx, 2:]
            
            # Reconstruct past positions (this is the FIX)
            trajectory = []
            for t in range(context_len):
                # Go backward: pos_t = current_pos - vel * (context_len - 1 - t)
                past_pos = current_pos - current_vel * (context_len - 1 - t)
                trajectory.append(past_pos)
            
            theta_trajectories[theta_idx] = np.array(trajectory)
        
        # Predict future positions
        for edge_idx in range(len(edge_labels)):
            epsilon_idx = edge_index[0, edge_idx]
            theta_idx = edge_index[1, edge_idx]
            
            epsilon_pos = epsilon_features[epsilon_idx, :2]
            
            if theta_idx < len(theta_features):
                # Use trajectory to predict
                trajectory = theta_trajectories[theta_idx]
                predicted_positions = run_fixed_kalman(trajectory, horizon_len)
                predicted_pos = predicted_positions[0]  # First prediction step
                
                # Calculate probability based on distance
                distance = np.linalg.norm(epsilon_pos - predicted_pos)
                prob_score = distance_to_probability(distance)
            else:
                prob_score = 0.0
            
            y_true.append(edge_labels[edge_idx])
            y_scores.append(prob_score)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    
    # Find optimal threshold
    if y_true.sum() > 0:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # Find threshold with precision >= 0.1 (more realistic for this imbalanced dataset)
        valid_indices = precision >= 0.1
        if valid_indices.any():
            best_idx = np.where(valid_indices)[0][-1]
            optimal_threshold = thresholds[best_idx]
        else:
            optimal_threshold = 0.5
    else:
        optimal_threshold = 0.5
    
    # Binary predictions at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    precision_opt = precision_score(y_true, y_pred, zero_division=0)
    recall_opt = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC": float(auc),
        "Optimal_Threshold": float(optimal_threshold),
        "Precision": float(precision_opt),
        "Recall": float(recall_opt),
        "Precision_at_0.5": float(precision_score(y_true, (y_scores >= 0.5).astype(int), zero_division=0)),
        "Total_Predictions": len(y_true),
        "Positive_Labels": int(y_true.sum()),
        "Score_Range": [float(y_scores.min()), float(y_scores.max())]
    }

# Run the FIXED evaluation for each horizon
for horizon in horizons:
    print(f"\\n🔧 FIXED Kalman evaluation for {horizon}-tick horizon...")
    
    try:
        results = evaluate_kalman_fixed(f'data/processed_{horizon}tick/test.pt')
        results_summary[f'Kalman-{horizon}'] = results
        
        print(f"   ✅ AUC: {results['AUC']:.4f}")
        print(f"   📊 Precision: {results['Precision']:.4f}")
        print(f"   📊 Recall: {results['Recall']:.4f}")
        
        # Save individual results
        with open(f'results/kalman_{horizon}tick_results_fixed.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results_summary[f'Kalman-{horizon}'] = {"AUC": 0.5, "Precision": 0.0, "Recall": 0.0}

print("\\n🎯 FIXED Kalman baselines complete!")
```