# 🚀 Quick Command Reference

> ⚠️ **Critical**: Models predict future detections ONLY for entities already detected in context window. See [docs/technical/prediction_protocol.md](docs/technical/prediction_protocol.md) for details.

## Graph Building (After raw data generation)

### Option 1: Build all datasets at once (FAST - Parallel)
```bash
./scripts/build_all_graphs_parallel.sh  # ~30 minutes for all horizons
```

### Option 2: Build individually (SLOW - Sequential)
⚠️ **WARNING**: The commands below use sequential processing and can take 10+ hours!
Use the parallel script above instead.
```bash
# Easy task (30 ticks = 64 seconds)
python -m aquascan.dataset.build_graphs \
    --raw data/raw_5tick \
    --out data/processed_30tick \
    --context 60 --horizon 30 \
    --split 0.7 0.15 0.15

# Moderate task (100 ticks = 3.5 minutes)
python -m aquascan.dataset.build_graphs \
    --raw data/raw_5tick \
    --out data/processed_100tick \
    --context 60 --horizon 100 \
    --split 0.7 0.15 0.15

# Challenging task (150 ticks = 5.3 minutes)
python -m aquascan.dataset.build_graphs \
    --raw data/raw_5tick \
    --out data/processed_150tick \
    --context 60 --horizon 150 \
    --split 0.7 0.15 0.15
```

## Upload to Colab
```bash
# Compress for upload
tar -czf aquascan_graphs.tar.gz data/processed_*tick/

# Or individually
tar -czf graphs_30tick.tar.gz data/processed_30tick/
tar -czf graphs_100tick.tar.gz data/processed_100tick/
tar -czf graphs_150tick.tar.gz data/processed_150tick/
```

## On Colab
```python
# Extract
!tar -xzf aquascan_graphs.tar.gz

# Install deps
!pip install torch torch_geometric scikit-learn

# Train GNN
!python scripts/gnn_train.py \
    --data data/processed_30tick/train.pt \
    --val data/processed_30tick/val.pt \
    --epochs 100

# Evaluate
!python scripts/gnn_eval.py \
    --ckpt checkpoints/gnn_30tick.pt \
    --data data/processed_30tick/test.pt
```
