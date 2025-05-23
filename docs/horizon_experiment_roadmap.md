# 🧪 Experiment Roadmap: Testing Task Difficulty Hypothesis

## 🎯 **Hypothesis to Test**
The GNN's "perfect" AUC = 1.00 performance is due to **trivially short prediction horizon (30 ticks = 30 seconds)**. Increasing to **300 ticks (5 minutes)** should make the task realistically challenging and reduce GNN performance to believable levels.

## 📋 **Complete Experimental Protocol**

### **Phase 1: Modify Dataset Generation** ⏱️ ~30 min

#### 1.1 Update Graph Builder Parameters
```bash
# Edit: aquascan/dataset/build_graphs.py
# Change default parameters from:
context_len=60, horizon_len=30
# To:
context_len=60, horizon_len=300  # 5-minute prediction horizon
```

**Note**: Key simulation files are located at:
- `aquascan/run_simulation.py` - Main entry point (not in root)
- `aquascan/simulation/simulation_loop.py` - Core simulation loop
- `aquascan/simulation/entities.py` - Entity definitions

#### 1.2 Create New Configuration
```bash
# Create: configs/long_horizon.yml
sim:
  ticks: 1200           # Need longer simulations (20 minutes)
  seed: 42
  resolution_km: 1.0
  
dataset:
  context_len: 60       # Keep same context (1 minute)
  horizon_len: 300      # 5-minute prediction horizon
  
bokeh:
  port: 5006
  show: false
```

#### 1.3 Backup Current Results
```bash
# Backup existing "perfect" results
mkdir -p results/backup_30tick_experiment
cp results/*.json results/backup_30tick_experiment/
cp -r data/processed data/processed_30tick_backup
```

### **Phase 2: Generate New Dataset** ⏱️ ~2-4 hours

#### 2.1 Generate Raw Simulation Data
```bash
# Generate longer simulations (need 1200+ ticks for 300-tick horizon)
python -m aquascan.batch.generate \
    --cfg configs/long_horizon.yml \
    --runs 100 \
    --out data/raw_300tick \
    --jobs 8
```

NOTE: We want to generate the data on x128, a 2 minute simulation for config is enough.

#### 2.2 Build Graph Dataset with Long Horizon
```bash
# Process into graphs with 300-tick horizon
python -m aquascan.dataset.build_graphs \
    --raw data/raw_300tick \
    --out data/processed_300tick \
    --context 60 \
    --horizon 300 \
    --split 0.7 0.15 0.15
```

### **Phase 3: Retrain Models** ⏱️ ~1-2 hours

#### 3.1 Train New GNN Model
```bash
# Train on new challenging dataset
python scripts/gnn_train.py \
    --data data/processed_300tick/train.pt \
    --val data/processed_300tick/val.pt \
    --epochs 100 \
    --lr 1e-3 \
    --batch 8 \
    --out checkpoints/gnn_300tick.pt
```

#### 3.2 Evaluate New GNN
```bash
# Test on 300-tick horizon data
python scripts/gnn_eval.py \
    --ckpt checkpoints/gnn_300tick.pt \
    --data data/processed_300tick/test.pt \
    --out results/gnn_300tick_test.json
```

### **Phase 4: Update Kalman Baseline** ⏱️ ~30 min

#### 4.1 Modify Kalman Evaluator
```bash
# Edit: scripts/kalman_eval.py
# Update to handle 300-tick prediction horizon
# Key changes:
# - Predict 300 steps ahead instead of 30
# - Adjust motion model uncertainty for longer horizon
```

#### 4.2 Run Kalman on New Data
```bash
# Evaluate Kalman on 300-tick task
python scripts/kalman_eval.py \
    --data data/processed_300tick/test.pt \
    --horizon 300 \
    --out results/kalman_300tick_test.json
```

### **Phase 5: Compare Results** ⏱️ ~15 min

#### 5.1 Generate Comparison
```bash
# Compare 30-tick vs 300-tick results
python scripts/compare_horizons.py \
    --results_30tick results/backup_30tick_experiment/ \
    --results_300tick results/ \
    --out results/horizon_comparison.json
```

#### 5.2 Update Evaluation Package
```bash
# Regenerate all evaluation artifacts
python scripts/make_metrics_table.py    # Should show realistic GNN performance
python scripts/plot_curves.py           # Should show non-perfect curves  
python scripts/make_snapshots.py        # Update with new predictions
```

## 📊 **Expected Results**

### **Hypothesis Confirmed (Task was too easy):**
| Model | 30-tick AUC | 300-tick AUC | Interpretation |
|-------|-------------|---------------|----------------|
| Kalman | 0.80 | 0.65-0.70 | Harder task, lower performance |
| GNN | 1.00 🚨 | 0.85-0.92 ✅ | Realistic performance, still better |

### **Hypothesis Rejected (Something else wrong):**
| Model | 30-tick AUC | 300-tick AUC | Interpretation |
|-------|-------------|---------------|----------------|
| Kalman | 0.80 | 0.65-0.70 | Expected degradation |
| GNN | 1.00 🚨 | 1.00 🚨 | Still perfect → other issue |

## ⚠️ **Potential Challenges**

### **Technical Issues:**
1. **Memory**: Longer horizons = more edges per graph → higher memory usage
2. **Training time**: More complex task may need longer training
3. **Data sparsity**: Longer horizons may have fewer positive examples

### **Fixes if Needed:**
```bash
# If memory issues:
--batch 4    # Reduce batch size

# If training convergence issues:
--epochs 200 --lr 5e-4    # More epochs, lower learning rate

# If too few positives:
--runs 200   # Generate more simulation runs
```

## 🎯 **Success Criteria**

### **Experiment is Successful if:**
✅ GNN AUC drops from 1.00 to 0.85-0.95 range  
✅ Both models show performance degradation (harder task)  
✅ Performance curves show realistic shapes (not perfect)  
✅ GNN still outperforms Kalman, but believably  

### **Experiment Reveals Other Issues if:**
❌ GNN still achieves AUC = 1.00 (suggests information leakage)  
❌ Both models perform identically (suggests task still trivial)  
❌ Models perform worse than random (suggests broken setup)  

## 📁 **File Organization**

```
# New directory structure
aquascan-gnns/
├── configs/
│   └── long_horizon.yml          # New config
├── data/
│   ├── raw_300tick/              # New raw data
│   ├── processed_300tick/        # New processed data
│   └── processed_30tick_backup/  # Backup original
├── results/
│   ├── backup_30tick_experiment/ # Original results
│   ├── gnn_300tick_test.json     # New GNN results
│   ├── kalman_300tick_test.json  # New Kalman results
│   └── horizon_comparison.json   # Comparison analysis
└── checkpoints/
    └── gnn_300tick.pt           # New trained model
```

## ⏱️ **Total Time Estimate: 4-7 hours**
- Dataset generation: 2-4 hours (depends on CPU cores)
- Model training: 1-2 hours  
- Evaluation & analysis: 1 hour
- Buffer for debugging: 1 hour

This experiment will definitively answer whether the "perfect" performance is due to task triviality or indicates a deeper issue! 🧪
