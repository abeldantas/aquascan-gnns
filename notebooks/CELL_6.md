## Cell 6: Train GNN Models (Fixed Arguments)

```python
# Train GNN for each horizon
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🎮 Training on: {device}")

# IMPORTANT: Set PYTHONPATH for imports
import os
os.environ['PYTHONPATH'] = '/content/aquascan-gnns'

results_summary = {}

for horizon in horizons:
    print(f"\\n{'='*60}")
    print(f"🧠 Training GNN for {horizon}-tick horizon...")
    print(f"{'='*60}")
    
    # Create results directory
    !mkdir -p /content/aquascan-gnns/results
    !mkdir -p /content/aquascan-gnns/checkpoints
    
    # Train the model (it expects train.pt and val.pt in the data directory)
    !cd /content/aquascan-gnns && python -m scripts.gnn_train \
        --data data/processed_{horizon}tick \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --patience 5
    
    # The script saves to checkpoints/best.pt, so rename it
    !cd /content/aquascan-gnns && mv checkpoints/best.pt checkpoints/gnn_{horizon}tick.pt
    
    # Evaluate (it takes --ckpt not --checkpoint)
    print(f"\\n📊 Evaluating GNN...")
    !cd /content/aquascan-gnns && python -m scripts.gnn_eval \
        --ckpt checkpoints/gnn_{horizon}tick.pt \
        --data data/processed_{horizon}tick/test.pt
    
    # The script saves to results/gnn_test.json, so rename it
    !cd /content/aquascan-gnns && mv results/gnn_test.json results/gnn_{horizon}tick_results.json
    
    # Load results - use full path
    results_file = f'/content/aquascan-gnns/results/gnn_{horizon}tick_results.json'
    if os.path.exists(results_file):
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
            results_summary[f'GNN-{horizon}'] = results
            print(f"\\n✅ Results: AUC={results['AUC']:.4f}")
            print(f"   Precision@optimal: {results['Precision_at_Optimal']:.4f}")
            print(f"   Recall@optimal: {results['Recall_at_Optimal']:.4f}")
    else:
        print(f"\\n⚠️ No results file found for {horizon}-tick horizon")
        results_summary[f'GNN-{horizon}'] = {"AUC": 0, "Precision": 0, "Recall": 0}

# Show summary
print(f"\\n{'='*60}")
print("📊 TRAINING SUMMARY")
print(f"{'='*60}")
for key, val in results_summary.items():
    if 'AUC' in val:
        print(f"{key}: AUC={val['AUC']:.4f}")
```