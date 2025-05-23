# 🌊 Aquascan Full Pipeline - Colab Pro Edition

This notebook runs the ENTIRE Aquascan pipeline on Colab Pro, from raw data generation to trained models.

## Cell 1: Environment Setup

```python
# Check GPU and mount Drive
import os
import sys
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Check resources
!nvidia-smi
!cat /proc/cpuinfo | grep 'model name' | uniq
!free -h

# Create working directory
WORK_DIR = "/content/aquascan-gnns"
!mkdir -p {WORK_DIR}
%cd {WORK_DIR}

print("✅ Environment ready!")
```

## Cell 2: Upload and Extract Code

```python
# Option A: Upload from local machine
from google.colab import files
uploaded = files.upload()  # Select aquascan-colab.tar.gz

# Extract
!tar -xzf aquascan-colab.tar.gz --strip-components=1
!ls -la

# Option B: If you uploaded to Drive first
# !cp /content/drive/MyDrive/aquascan-colab.tar.gz .
# !tar -xzf aquascan-colab.tar.gz --strip-components=1
```

## Cell 3: Install Dependencies

```python
# Install requirements
!pip install -r requirements.txt

# Additional packages for parallel processing
!pip install tqdm joblib

# Verify key imports
import torch
import h5py
import numpy as np
from tqdm import tqdm
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
```

## Cell 4: Generate Raw Simulation Data (The Heavy Part)

```python
# OPTION 1: Quick test (10 runs)
!python -m aquascan.batch.generate \
    --cfg configs/optimal_5tick.yml \
    --runs 10 \
    --out data/raw_5tick \
    --jobs -1  # Use all available cores

# OPTION 2: Medium dataset (100 runs) 
!python -m aquascan.batch.generate \
    --cfg configs/optimal_5tick.yml \
    --runs 100 \
    --out data/raw_5tick \
    --jobs -1

# OPTION 3: Full dataset (1000 runs) - Takes ~2-3 hours
!python -m aquascan.batch.generate \
    --cfg configs/optimal_5tick.yml \
    --runs 1000 \
    --out data/raw_5tick \
    --jobs -1
```

## Cell 5: Monitor Progress

```python
# Check progress in real-time
import time
import glob

def monitor_generation():
    while True:
        files = glob.glob('data/raw_5tick/*.h5')
        size_mb = sum(os.path.getsize(f) for f in files) / (1024*1024)
        print(f"\r📊 Files: {len(files)} | Size: {size_mb:.1f} MB", end='')
        time.sleep(5)
        
# Run this in a separate cell while generation is running
# monitor_generation()
```

## Cell 6: Build Graph Datasets (All Horizons)

```python
# Use the parallel builder
import subprocess
import multiprocessing

# Detect available cores
cores = multiprocessing.cpu_count()
print(f"🚀 Using {cores} CPU cores")

# Build all three horizons
horizons = [30, 100, 150]
for horizon in horizons:
    print(f"\n{'='*60}")
    print(f"Building {horizon}-tick horizon dataset...")
    print(f"{'='*60}")
    
    subprocess.run([
        'python', 'scripts/build_graphs_parallel.py',
        '--raw', 'data/raw_5tick',
        '--out', f'data/processed_{horizon}tick',
        '--context', '60',
        '--horizon', str(horizon),
        '--split', '0.7', '0.15', '0.15',
        '--jobs', str(cores)
    ])

print("\n✅ All graph datasets built!")
!du -sh data/processed_*
```

## Cell 7: Train Models (GPU Accelerated)

```python
# Train GNN for each horizon
horizons = [30, 100, 150]

for horizon in horizons:
    print(f"\n{'='*60}")
    print(f"Training GNN for {horizon}-tick horizon...")
    print(f"{'='*60}")
    
    !python scripts/gnn_train.py \
        --data data/processed_{horizon}tick/train.pt \
        --val data/processed_{horizon}tick/val.pt \
        --epochs 100 \
        --batch 32 \
        --lr 1e-3 \
        --device cuda \
        --checkpoint checkpoints/gnn_{horizon}tick.pt
    
    # Evaluate immediately
    !python scripts/gnn_eval.py \
        --checkpoint checkpoints/gnn_{horizon}tick.pt \
        --data data/processed_{horizon}tick/test.pt \
        --device cuda \
        --output results/gnn_{horizon}tick_results.json
```

## Cell 8: Run Kalman Baselines

```python
# Run Kalman filter baseline for comparison
for horizon in [30, 100, 150]:
    print(f"\nEvaluating Kalman filter for {horizon}-tick horizon...")
    
    # Modify the kalman script to accept horizon parameter
    !python scripts/kalman_eval.py \
        --data data/processed_{horizon}tick/test.pt \
        --output results/kalman_{horizon}tick_results.json
```

## Cell 9: Compare Results

```python
import json
import pandas as pd

# Load all results
results = []
for model in ['gnn', 'kalman']:
    for horizon in [30, 100, 150]:
        with open(f'results/{model}_{horizon}tick_results.json', 'r') as f:
            data = json.load(f)
            data['Model'] = model.upper()
            data['Horizon'] = horizon
            results.append(data)

# Create comparison table
df = pd.DataFrame(results)
df = df.pivot_table(index='Model', columns='Horizon', values=['AUC', 'Precision', 'Recall'])
print("\n📊 RESULTS COMPARISON")
print("="*60)
print(df.round(4))

# Plot results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['AUC', 'Precision', 'Recall']

for i, metric in enumerate(metrics):
    ax = axes[i]
    for model in ['GNN', 'KALMAN']:
        model_data = df.loc[model, metric]
        ax.plot([30, 100, 150], model_data, 'o-', label=model, linewidth=2, markersize=8)
    
    ax.set_xlabel('Prediction Horizon (ticks)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs Prediction Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/horizon_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Cell 10: Save Everything to Drive

```python
# Create archive of results
!tar -czf aquascan_results.tar.gz data/processed_* results/ checkpoints/

# Copy to Drive
!cp aquascan_results.tar.gz /content/drive/MyDrive/

# Also save raw data if you want to keep it
!tar -czf aquascan_raw_data.tar.gz data/raw_5tick/
!cp aquascan_raw_data.tar.gz /content/drive/MyDrive/

print("✅ All results saved to Google Drive!")
print("📦 Results: /content/drive/MyDrive/aquascan_results.tar.gz")
print("📦 Raw data: /content/drive/MyDrive/aquascan_raw_data.tar.gz")
```

## Cell 11: Quick Debugging Tools

```python
# Useful debugging commands

# Check a sample HDF5 file
def inspect_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        print(f"Nodes shape: {f['nodes'].shape}")
        print(f"Edges shape: {f['edges'].shape}")
        
# Check a sample graph
def inspect_graph(filepath):
    import torch
    graphs = torch.load(filepath)
    g = graphs[0]
    print(f"Loaded {len(graphs)} graphs")
    print(f"Sample graph:")
    print(f"  Epsilon nodes: {g['epsilon'].x.shape}")
    print(f"  Theta nodes: {g['theta'].x.shape}")
    print(f"  Detection targets: {g['epsilon', 'will_detect', 'theta'].edge_label.shape}")

# Memory usage
def check_memory():
    !free -h
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB used")
```

## Pro Tips for Colab

1. **Session Management**: 
   - Use `Ctrl+Shift+P` → "Prevent session timeout" 
   - Keep the tab active to prevent disconnection

2. **Checkpointing**:
   - Save intermediate results to Drive frequently
   - The notebook auto-saves but your data doesn't!

3. **Resource Monitoring**:
   - Check GPU usage: `!nvidia-smi -l 1`
   - Check CPU: `!htop` (install with `!apt install htop`)

4. **Parallel Processing**:
   - Colab Pro usually gives 2-8 CPUs
   - Adjust `--jobs` parameter accordingly

5. **Storage**:
   - Colab disk is temporary (~100GB)
   - Always save important results to Drive

## Estimated Runtime (Colab Pro)

- **Raw data generation (1000 runs)**: ~2-3 hours
- **Graph building (all horizons)**: ~45-60 minutes  
- **Model training (all horizons)**: ~30-45 minutes
- **Total**: ~4-5 hours

Much better than your Mac melting for 12+ hours! 🚀
