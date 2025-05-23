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

# Install PyTorch Geometric (special handling required)
import torch
print(f"\\n🔧 Installing PyTorch Geometric for PyTorch {torch.__version__}...")

# Get CUDA version and install appropriate wheels
if torch.cuda.is_available():
    cuda_version = torch.version.cuda.replace('.', '')
    torch_version = torch.__version__.split('+')[0]
    cuda_tag = f"cu{cuda_version[:3]}"
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Installing for: torch-{torch_version}+{cuda_tag}")
    !pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html
else:
    print(f"   Installing CPU version...")
    !pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{torch.__version__}.html

# Install torch_geometric
!pip install -q torch-geometric

# Verify installation
try:
    import torch_geometric
    print(f"\\n✅ PyTorch Geometric {torch_geometric.__version__} installed!")
except:
    print("\\n❌ PyTorch Geometric installation failed! Trying alternative method...")
    !bash scripts/install_torch_geometric_colab.sh

print(f"\\n✅ Setup complete!")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# Check available resources
print("\\n📊 System Resources:")
!free -h
!df -h /content
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

## Cell 5: Build Graph Datasets (Fixed for Colab)

```python
# Fix shared memory issue first
print("🔧 Fixing shared memory for PyTorch multiprocessing...")
!df -h /dev/shm
!sudo mount -o remount,size=10G /dev/shm || echo "Could not increase shared memory"
!df -h /dev/shm

# Alternative: Use file system strategy if shared memory is limited
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
print("✅ Using file_system strategy for tensor sharing")

# Build graphs for all three horizons
import time
import multiprocessing

horizons = [30, 100, 150]
horizon_names = ['easy (64s)', 'moderate (3.5min)', 'challenging (5.3min)']

print(f"\\n🔨 Building graph datasets for {len(horizons)} prediction horizons...")
print(f"💪 You have 50GB RAM - let's use it!")

# Get CPU count
CPUS = multiprocessing.cpu_count()
print(f"⚡ Using {CPUS} CPU cores\\n")

total_start = time.time()

for horizon, name in zip(horizons, horizon_names):
    print(f"\\n{'='*60}")
    print(f"📈 Horizon: {horizon} ticks - {name}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Run parallel graph builder with file system strategy
    !cd /content/aquascan-gnns && PYTORCH_SHARE_STRATEGY=file_system python scripts/build_graphs_parallel.py \
        --raw data/raw_5tick \
        --out data/processed_{horizon}tick \
        --context 60 \
        --horizon {horizon} \
        --split 0.7 0.15 0.15 \
        --jobs {CPUS}
    
    # Verify it created the files
    print("\\nChecking created files:")
    !ls -la data/processed_{horizon}tick/
    
    elapsed = time.time() - start
    print(f"\\n✅ {horizon}-tick dataset complete in {elapsed/60:.1f} minutes")
    
    # Clear memory between horizons
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

total_elapsed = time.time() - total_start
print(f"\\n🎉 All graph datasets built in {total_elapsed/60:.1f} minutes!")

# Check sizes
print("\\n📦 Dataset sizes:")
!du -sh data/processed_*

# Verify the splits were created
print("\\n📁 Files created:")
for horizon in horizons:
    print(f"\\n{horizon}-tick horizon:")
    !ls -lh data/processed_{horizon}tick/*.pt
```

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

## Cell 7: Run Kalman Baselines

```python
# Run Kalman filter baseline for comparison
print("📐 Running Kalman filter baselines...\\n")

# The kalman script expects a single test.pt file path
for horizon in horizons:
    print(f"\\nEvaluating Kalman for {horizon}-tick horizon...")
    
    # Create a temporary script that accepts our arguments
    kalman_wrapper = f'''
import sys
sys.path.append('/content/aquascan-gnns')
import os
os.environ['PYTHONPATH'] = '/content/aquascan-gnns'

# Import the kalman evaluation function
from scripts.kalman_eval import evaluate_file
import json

# Run evaluation
results = evaluate_file('data/processed_{horizon}tick/test.pt')

# Save results
with open('results/kalman_{horizon}tick_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results: AUC={{results['AUC']:.4f}}")
'''
    
    # Write and run the wrapper
    with open(f'kalman_wrapper_{horizon}.py', 'w') as f:
        f.write(kalman_wrapper)
    
    !cd /content/aquascan-gnns && python kalman_wrapper_{horizon}.py
    
    # Load results
    with open(f'results/kalman_{horizon}tick_results.json', 'r') as f:
        results = json.load(f)
        results_summary[f'Kalman-{horizon}'] = results

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
        if metric in ['AUC', 'Precision', 'Recall']:  # Filter to standard metrics
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
        if len(model_data) > 0:
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

# Key findings
print("\\n🔍 KEY FINDINGS:")
for horizon in horizons:
    gnn_auc = df[(df['Model']=='GNN') & (df['Horizon']==horizon) & (df['Metric']=='AUC')]['Value'].values
    kalman_auc = df[(df['Model']=='Kalman') & (df['Horizon']==horizon) & (df['Metric']=='AUC')]['Value'].values
    if len(gnn_auc) > 0 and len(kalman_auc) > 0:
        improvement = ((gnn_auc[0] - kalman_auc[0]) / kalman_auc[0]) * 100
        print(f"  - {horizon}-tick: GNN improves AUC by {improvement:.1f}% over Kalman")
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
!cd /content/aquascan-gnns && tar -czf aquascan_results_{timestamp}.tar.gz results/ checkpoints/
!cd /content/aquascan-gnns && tar -czf aquascan_graphs_{timestamp}.tar.gz data/processed_*

# Copy to Drive
!mkdir -p /content/drive/MyDrive/aquascan_backups
!cp /content/aquascan-gnns/aquascan_results_{timestamp}.tar.gz /content/drive/MyDrive/aquascan_backups/
!cp /content/aquascan-gnns/aquascan_graphs_{timestamp}.tar.gz /content/drive/MyDrive/aquascan_backups/

print(f"\\n✅ Results backed up to Google Drive!")
print(f"📁 /content/drive/MyDrive/aquascan_backups/aquascan_results_{timestamp}.tar.gz")
print(f"📁 /content/drive/MyDrive/aquascan_backups/aquascan_graphs_{timestamp}.tar.gz")
```

## Cell 10: Push Results to GitHub

```python
# Commit and push results to GitHub
print("🚀 Pushing results to GitHub...\\n")

# Go to repo directory
%cd /content/aquascan-gnns

# Check what we're committing
!git status

# Add results and visualizations
!git add results/*.json
!git add results/*.png
!git add -f checkpoints/*.pt 2>/dev/null || echo "Checkpoints might be too large for git"

# Create detailed commit message
if len(results_summary) > 0:
    # Find best GNN result
    gnn_results = [(h, results_summary[f'GNN-{h}']['AUC']) for h in horizons if f'GNN-{h}' in results_summary]
    if gnn_results:
        best_gnn = max(gnn_results, key=lambda x: x[1])
        commit_msg = f"🤖 Colab run: GNN AUC={best_gnn[1]:.4f} @ {best_gnn[0]}t horizon"
        
        # Get performance drop info
        if len(gnn_results) >= 2:
            perf_30 = next((r[1] for r in gnn_results if r[0] == 30), 0)
            perf_150 = next((r[1] for r in gnn_results if r[0] == 150), 0)
            if perf_30 > 0:
                drop = (perf_30 - perf_150) / perf_30 * 100
                details = f"Performance drop from 30t to 150t: {drop:.1f}%"
            else:
                details = "Complete results in results/"
        else:
            details = "Partial run - see results/"
    else:
        commit_msg = "🤖 Colab run completed"
        details = "Results in results/"
else:
    commit_msg = "🤖 Colab pipeline test"
    details = "Testing pipeline"

# Commit
!git commit -m "{commit_msg}" -m "{details}" -m "Run on Colab Pro with {NUM_RUNS} simulations" || echo "Nothing to commit"

# Push
!git push origin main || echo "Push failed - check your token permissions"

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
    if f'GNN-{horizon}' in results_summary and f'Kalman-{horizon}' in results_summary:
        gnn_auc = results_summary[f'GNN-{horizon}']['AUC']
        kalman_auc = results_summary[f'Kalman-{horizon}']['AUC']
        improvement = ((gnn_auc - kalman_auc) / kalman_auc) * 100
        print(f"  {horizon:3d}-tick horizon: GNN={gnn_auc:.4f}, Kalman={kalman_auc:.4f} (+{improvement:.1f}%)")
    else:
        print(f"  {horizon:3d}-tick horizon: Missing results")

print(f"\\n🔍 Hypothesis Test:")
if 'GNN-30' in results_summary and 'GNN-150' in results_summary:
    if results_summary['GNN-30']['AUC'] > 0.95 and results_summary['GNN-150']['AUC'] < 0.90:
        print("  ✅ CONFIRMED: Short horizons are too easy, longer horizons provide appropriate challenge")
    else:
        print("  ❓ RESULTS: Performance pattern doesn't match initial hypothesis")
        print(f"     30-tick: {results_summary['GNN-30']['AUC']:.4f}")
        print(f"     150-tick: {results_summary['GNN-150']['AUC']:.4f}")

print(f"\\n📁 Artifacts saved:")
print(f"  - Models: checkpoints/gnn_*tick.pt")
print(f"  - Results: results/*_results.json") 
print(f"  - Plots: results/horizon_comparison.png")
print(f"  - Backups: Google Drive /aquascan_backups/")

print(f"\\n⏱️  Total runtime: {(time.time() - total_start)/3600:.1f} hours")
```

## Troubleshooting

### Common Issues and Fixes:

1. **PyTorch Geometric Installation Fails**
   ```python
   # Manual installation
   !pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3])").html
   !pip install torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3])").html
   !pip install torch-geometric
   ```

2. **Shared Memory Error (RuntimeError: unable to mmap)**
   ```python
   # Already included in Cell 5, but if needed:
   import torch.multiprocessing
   torch.multiprocessing.set_sharing_strategy('file_system')
   ```

3. **Out of Memory During Training**
   ```python
   # Reduce batch size in Cell 6
   --batch-size 16  # Instead of 32
   ```

4. **Git Push Fails**
   - Check your token has 'repo' permissions
   - Try: `!git push https://$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$GITHUB_REPO.git main`

5. **Module Not Found Errors**
   ```python
   # Always set these at the start of cells that run scripts
   import os
   os.environ['PYTHONPATH'] = '/content/aquascan-gnns'
   ```

## Next Steps

After running this pipeline:

1. **Download key files locally:**
   ```python
   from google.colab import files
   files.download('/content/aquascan-gnns/results/horizon_comparison.png')
   files.download('/content/aquascan-gnns/checkpoints/gnn_150tick.pt')
   ```

2. **Check your GitHub repo** for the pushed results

3. **Review the Google Drive backups** for full datasets

Happy modeling! 🌊🤖
