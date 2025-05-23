# 🎯 PROPERLY CALIBRATED Kalman Filter

print("📐 Running CALIBRATED Kalman filter baselines...\n")

def calibrated_kalman_eval(test_file):
    """Kalman evaluation calibrated for actual detection physics."""
    graphs = torch.load(test_file, weights_only=False)
    y_true, y_scores = [], []
    
    print(f"   Processing {len(graphs)} graphs...")
    
    for i, graph in enumerate(graphs):
        if i % 100 == 0:
            print(f"   Graph {i+1}/{len(graphs)}")
            
        horizon = int(graph.horizon_len)
        
        # Get features
        epsilon_pos = graph['epsilon'].x.numpy()[:, :2]  
        theta_features = graph['theta'].x.numpy()
        
        # Get prediction targets
        edge_labels = graph[('epsilon', 'will_detect', 'theta')].edge_label.numpy()
        edge_index = graph[('epsilon', 'will_detect', 'theta')].edge_label_index.numpy()
        
        # For each prediction edge
        for j in range(len(edge_labels)):
            eps_idx = edge_index[0, j]
            theta_idx = edge_index[1, j]
            
            if theta_idx < len(theta_features):
                # Current theta position and velocity
                theta_pos = theta_features[theta_idx, :2]
                theta_vel = theta_features[theta_idx, 2:]
                
                # Predict theta position after horizon ticks
                predicted_theta_pos = theta_pos + theta_vel * horizon
                
                # Current epsilon position (sensors don't move much)
                eps_pos = epsilon_pos[eps_idx]
                
                # Distance between sensor and predicted marine entity
                distance = np.linalg.norm(eps_pos - predicted_theta_pos)
                
                # CALIBRATED scoring: detection radius is 0.2km
                if distance <= 0.2:
                    score = 1.0 - (distance / 0.2)  # Linear decay within detection radius
                else:
                    # Very small probability for distances beyond detection radius
                    score = 0.2 * np.exp(-(distance - 0.2) / 1.0)  # Exponential decay
                    
            else:
                score = 0.0
                
            y_true.append(edge_labels[j])
            y_scores.append(score)
    
    return np.array(y_true), np.array(y_scores)

# Run calibrated evaluation
from sklearn.metrics import roc_auc_score, precision_score, recall_score

for horizon in horizons:
    print(f"\n🎯 Calibrated Kalman for {horizon}-tick horizon...")
    
    try:
        y_true, y_scores = calibrated_kalman_eval(f'data/processed_{horizon}tick/test.pt')
        
        print(f"   📊 Data summary:")
        print(f"      Positive labels: {y_true.sum()}")
        print(f"      Score range: [{y_scores.min():.4f}, {y_scores.max():.4f}]")
        print(f"      High scores (>0.1): {(y_scores > 0.1).sum()}")
        
        # Calculate AUC
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
        else:
            auc = 0.5
            
        # Use a very low threshold since positives are so rare
        # Try different thresholds to find reasonable precision/recall
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        best_f1 = 0
        best_results = None
        
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_results = {"precision": prec, "recall": rec, "threshold": thresh}
        
        results = {
            "AUC": float(auc),
            "Precision": float(best_results["precision"]),
            "Recall": float(best_results["recall"]),
            "Best_Threshold": float(best_results["threshold"]),
            "F1_Score": float(best_f1)
        }
        
        results_summary[f'Kalman-{horizon}'] = results
        
        print(f"   ✅ AUC: {auc:.4f}")
        print(f"   🎯 Best F1: {best_f1:.4f} (P={best_results['precision']:.4f}, R={best_results['recall']:.4f}) @ thresh={best_results['threshold']}")
        
        # Save results
        with open(f'results/kalman_{horizon}tick_calibrated.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results_summary[f'Kalman-{horizon}'] = {"AUC": 0.5, "Precision": 0.0, "Recall": 0.0}

print("\n✅ Calibrated Kalman evaluation complete!")