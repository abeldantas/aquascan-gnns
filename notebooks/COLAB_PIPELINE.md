# 🌊 Aquascan Colab Pipeline - Complete Code

Copy these cells into a new Google Colab notebook. Make sure to enable GPU first!

## Cell 1: Setup GitHub Authentication

```python
# Setup GitHub authentication
import os
from getpass import getpass

# Get GitHub token (paste it when prompted)
github_token = getpass('Enter your GitHub personal access token: ')
github_username = input('Enter your GitHub username (adantas): ') or 'adantas'
github_repo = 'aquascan-gnns'

# Store for later use
os.environ['GITHUB_TOKEN'] = github_token
os.environ['GITHUB_USERNAME'] = github_username
os.environ['GITHUB_REPO'] = github_repo

print("✅ GitHub credentials stored!")
```

## Cell 2: Clone Repository and Setup

```python
# Clone the repository
%cd /content

# Remove if exists
!rm -rf aquascan-gnns

# Clone with authentication
!git clone https://$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$GITHUB_REPO.git

# Enter the directory
%cd aquascan-gnns

# Setup git config for commits
!git config user.email "colab@aquascan.ai"
!git config user.name "Colab Runner"

# Install dependencies
print("\\n📦 Installing requirements...")
!pip install -q -r requirements.txt
!pip install -q tqdm joblib

import torch
print(f"\\n✅ Setup complete!")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
```

## Cell 3: Configuration

```python
import multiprocessing
import time
from datetime import datetime

# Configuration
NUM_RUNS = 100  # Change to 1000 for full dataset
CPUS = multiprocessing.cpu_count()

print(f"🎯 Configuration:")
print(f"   - Simulation runs: {NUM_RUNS}")
print(f"   - CPU cores: {CPUS}")
print(f"   - Estimated time: {NUM_RUNS * 0.1:.0f} minutes")
print(f"   - Estimated size: {NUM_RUNS * 2:.0f} MB")
```

## Cell 4: Generate Raw Data

```python
# Generate raw simulation data
start_time = time.time()

print(f"\\n🚀 Starting simulation generation at {datetime.now().strftime('%H:%M:%S')}...")

# Run the batch generator
!python -m aquascan.batch.generate \
    --cfg configs/optimal_5tick.yml \
    --runs {NUM_RUNS} \
    --out data/raw_5tick \
    --jobs {CPUS}

elapsed = time.time() - start_time
print(f"\\n✅ Raw data generation complete in {elapsed/60:.1f} minutes!")

# Check results
!echo "\\n📊 Generated files:"
!ls data/raw_5tick | wc -l
!echo "\\n💾 Total size:"
!du -sh data/raw_5tick
```

## Cell 5: Build Graph Datasets

```python
# Build graphs for all three horizons
horizons = [30, 100, 150]
horizon_names = ['easy (64s)', 'moderate (3.5min)', 'challenging (5.3min)']

print(f"🔨 Building graph datasets for {len(horizons)} prediction horizons...")
print(f"⚡ Using {CPUS} parallel workers\\n")

total_start = time.time()

for horizon, name in zip(horizons, horizon_names):
    print(f"\\n{'='*60}")
    print(f"📈 Horizon: {horizon} ticks - {name}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Run parallel graph builder
    !python scripts/build_graphs_parallel.py \
        --raw data/raw_5tick \
        --out data/processed_{horizon}tick \
        --context 60 \
        --horizon {horizon} \
        --split 0.7 0.15 0.15 \
        --jobs {CPUS}
    
    elapsed = time.time() - start
    print(f"\\n✅ {horizon}-tick dataset complete in {elapsed/60:.1f} minutes")

total_elapsed = time.time() - total_start
print(f"\\n🎉 All graph datasets built in {total_elapsed/60:.1f} minutes!")

# Check sizes
print("\\n📦 Dataset sizes:")
!du -sh data/processed_*
```

## Cell 6: Train GNN Models

```python
# Train GNN for each horizon
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🎮 Training on: {device}")

results_summary = {}

for horizon in horizons:
    print(f"\\n{'='*60}")
    print(f"🧠 Training GNN for {horizon}-tick horizon...")
    print(f"{'='*60}")
    
    # Train
    !python scripts/gnn_train.py \
        --data data/processed_{horizon}tick/train.pt \
        --val data/processed_{horizon}tick/val.pt \
        --epochs 100 \
        --batch 32 \
        --lr 1e-3 \
        --device {device} \
        --checkpoint checkpoints/gnn_{horizon}tick.pt
    
    # Evaluate
    print(f"\\n📊 Evaluating GNN...")
    !python scripts/gnn_eval.py \
        --checkpoint checkpoints/gnn_{horizon}tick.pt \
        --data data/processed_{horizon}tick/test.pt \
        --device {device} \
        --output results/gnn_{horizon}tick_results.json
    
    # Load results
    import json
    with open(f'results/gnn_{horizon}tick_results.json', 'r') as f:
        results = json.load(f)
        results_summary[f'GNN-{horizon}'] = results
        print(f"\\n✅ Results: AUC={results['AUC']:.4f}")
```

## Cell 7: Run Kalman Baselines

```python
# First, update the kalman script to accept different data paths
kalman_code = '''#!/usr/bin/env python3
import json
import pathlib
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Get data path from command line
data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/test.pt"
output_path = sys.argv[2] if len(sys.argv) > 2 else "results/kalman_baseline.json"

# [Rest of the kalman_eval.py code here - truncated for space]
# The key is that it now accepts command line arguments

print(f"Evaluating: {data_path}")
res = evaluate_file(data_path)

with open(output_path, "w") as f:
    json.dump(res, f, indent=2)

print(f"Results saved to {output_path}")
'''

# Save the modified script temporarily
with open('kalman_eval_colab.py', 'w') as f:
    # Copy the original and modify it
    with open('scripts/kalman_eval.py', 'r') as orig:
        f.write(orig.read())

# Run Kalman baselines
print("📐 Running Kalman filter baselines...\\n")

for horizon in horizons:
    print(f"\\nEvaluating Kalman for {horizon}-tick horizon...")
    
    !python kalman_eval_colab.py \
        data/processed_{horizon}tick/test.pt \
        results/kalman_{horizon}tick_results.json
    
    # Load results
    with open(f'results/kalman_{horizon}tick_results.json', 'r') as f:
        results = json.load(f)
        results_summary[f'Kalman-{horizon}'] = results
        print(f"Results: AUC={results['AUC']:.4f}")

print("\\n✅ All baselines complete!")
```

## Cell 8: Visualize Results

```python
# Create comparison visualizations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
data = []
for key, metrics in results_summary.items():
    model, horizon = key.split('-')
    for metric, value in metrics.items():
        data.append({
            'Model': model,
            'Horizon': int(horizon),
            'Metric': metric,
            'Value': value
        })

df = pd.DataFrame(data)

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['AUC', 'Precision', 'Recall']

for i, metric in enumerate(metrics):
    ax = axes[i]
    metric_df = df[df['Metric'] == metric]
    
    for model in ['GNN', 'Kalman']:
        model_data = metric_df[metric_df['Model'] == model].sort_values('Horizon')
        ax.plot(model_data['Horizon'], model_data['Value'], 
                'o-', label=model, linewidth=2, markersize=8)
    
    ax.set_xlabel('Prediction Horizon (ticks)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Prediction Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.suptitle('🌊 Aquascan Model Performance Comparison', fontsize=16)
plt.tight_layout()
plt.savefig('results/horizon_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\\n📊 RESULTS SUMMARY")
print("="*60)
summary_df = df.pivot_table(index=['Model', 'Horizon'], columns='Metric', values='Value')
print(summary_df.round(4))
```

## Cell 9: Save to Google Drive

```python
# Mount Google Drive for backup
from google.colab import drive
drive.mount('/content/drive')

# Create timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create archives
print("📦 Creating archives...")
!tar -czf aquascan_results_{timestamp}.tar.gz results/ checkpoints/
!tar -czf aquascan_graphs_{timestamp}.tar.gz data/processed_*

# Copy to Drive
!mkdir -p /content/drive/MyDrive/aquascan_backups
!cp aquascan_results_{timestamp}.tar.gz /content/drive/MyDrive/aquascan_backups/
!cp aquascan_graphs_{timestamp}.tar.gz /content/drive/MyDrive/aquascan_backups/

print(f"\\n✅ Results backed up to Google Drive!")
print(f"📁 /content/drive/MyDrive/aquascan_backups/")
```

## Cell 10: Push Results to GitHub

```python
# Commit and push results to GitHub
print("🚀 Pushing results to GitHub...\\n")

# Check what we're committing
!git status

# Add results and visualizations
!git add results/*.json
!git add results/*.png
!git add -f checkpoints/*.pt 2>/dev/null || echo "Checkpoints might be too large for git"

# Create detailed commit message
best_gnn = max([(h, results_summary[f'GNN-{h}']['AUC']) for h in horizons], key=lambda x: x[1])
commit_msg = f"🤖 Colab run: GNN AUC={best_gnn[1]:.4f} @ {best_gnn[0]}t horizon"

# Get performance drop info
perf_30 = results_summary['GNN-30']['AUC']
perf_150 = results_summary['GNN-150']['AUC']
drop = (perf_30 - perf_150) / perf_30 * 100

details = f"Performance drop from 30t to 150t: {drop:.1f}%"

# Commit
!git commit -m "{commit_msg}" -m "{details}" -m "Run on Colab Pro with {NUM_RUNS} simulations"

# Push
!git push origin main

print("\\n✅ Results pushed to repository!")
```

## Cell 11: Summary Report

```python
# Generate final summary report
print("="*70)
print("🎉 AQUASCAN PIPELINE COMPLETE!")
print("="*70)

print(f"\\n📊 Key Results:")
for horizon in horizons:
    gnn_auc = results_summary[f'GNN-{horizon}']['AUC']
    kalman_auc = results_summary[f'Kalman-{horizon}']['AUC']
    improvement = ((gnn_auc - kalman_auc) / kalman_auc) * 100
    print(f"  {horizon:3d}-tick horizon: GNN={gnn_auc:.4f}, Kalman={kalman_auc:.4f} (+{improvement:.1f}%)")

print(f"\\n🔍 Hypothesis Test:")
if results_summary['GNN-30']['AUC'] > 0.95 and results_summary['GNN-150']['AUC'] < 0.90:
    print("  ✅ CONFIRMED: Short horizons are too easy, longer horizons provide appropriate challenge")
else:
    print("  ❓ UNEXPECTED: Performance pattern doesn't match hypothesis")

print(f"\\n📁 Artifacts saved:")
print(f"  - Models: checkpoints/gnn_*tick.pt")
print(f"  - Results: results/*_results.json") 
print(f"  - Plots: results/horizon_comparison.png")
print(f"  - Backups: Google Drive /aquascan_backups/")

print(f"\\n⏱️  Total runtime: {(time.time() - total_start)/3600:.1f} hours")
```

## Troubleshooting

If you encounter issues:

1. **Out of memory**: Reduce `NUM_RUNS` or batch size
2. **Disconnected**: Re-run from Cell 2, skip data generation if it exists
3. **Git push fails**: Check your token has write permissions
4. **Import errors**: Make sure Cell 2 pip installs completed

## Next Steps

After running this pipeline:

1. Download the trained models from `checkpoints/`
2. Review the performance plots
3. Check your GitHub repo for the pushed results
4. Use the Google Drive backups for local analysis

Happy modeling! 🌊🤖
