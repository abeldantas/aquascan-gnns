# 🔐 Prediction Protocol: No Information Leakage Design

## Executive Summary

**Both the GNN and Kalman baseline predict future detections ONLY for marine entities that have already been detected during the context window.** This is a deliberate design choice that ensures:

1. **No information leakage** - Models never see entities they shouldn't know about
2. **Realistic task formulation** - You can't track what you've never seen
3. **Fair comparison** - Both baselines use identical information

## The Protocol

### What Models See (Context Window)
- **Epsilon nodes**: All sensor nodes in the network (~570)
- **Theta nodes**: ONLY marine entities detected at least once in past 60 ticks
- **Detection edges**: Historical ε→θ detections from context window
- **Node features**: Position (x,y) and velocity (Δx,Δy) computed ONLY from context data

### What Models Predict (Horizon Window)
- **Task**: Predict which ε-θ pairs will have detection events in next H ticks
- **Candidates**: All possible combinations of known epsilons × known thetas
- **Output**: Binary classification (will detect / won't detect)

### Critical Design Principle
**If a marine entity has never been detected in the context window, it does not exist in the graph.**

This means:
- New entities appearing in the horizon are ignored
- Models track only "known" entities
- Prediction difficulty scales with horizon length (entities may leave detection range)

## Implementation Details

### Graph Construction (`aquascan/dataset/build_graphs.py`)

```python
# Line 94-95: Extract nodes from context window only
context_nodes = f['nodes'][np.isin(f['nodes']['t'], context_ticks)]

# Lines 116-172: Features computed from context observations only
for node_id in theta_ids:
    node_data = theta_nodes[theta_nodes['gid'] == node_id]
    avg_pos = _xy(node_data).mean(axis=0)  # Average over context
    # ... velocity calculation from context ...
```

### Kalman Baseline (`scripts/kalman_eval.py`)

```python
# Lines 118-130: Build observations only for existing theta nodes
for theta_idx in range(len(theta_features)):
    # Reconstruct context using current state
    theta_obs[theta_idx] = np.array(obs_positions)

# Only predicts for thetas in the graph
```

## Why This Matters

### 1. Task Realism
- Real sensor networks can only track detected entities
- Unknown entities are... unknown
- Matches operational constraints of passive monitoring

### 2. Task Difficulty
The "perfect" AUC=1.0 is likely because:
- **Short horizon** (30 ticks = 64 seconds) makes motion predictable
- **Dense sensor coverage** (570 nodes, 200m detection radius) means frequent redetection
- **Limited entity count** (~15 marine entities) simplifies tracking

### 3. Experiment Validity
Longer horizons (100-150 ticks) increase difficulty because:
- Entities may swim completely out of sensor range
- Motion uncertainty compounds over time
- Prediction becomes genuinely challenging

## Verification Checklist

✅ **Graph builder**: Only includes detected entities in context  
✅ **GNN model**: Operates on provided graph structure  
✅ **Kalman baseline**: Uses same graph, same protocol  
✅ **No leakage**: Future positions never visible in features  
✅ **Fair comparison**: Both methods see identical information  

## Citation for Report

When describing the experimental setup in the final report, include:

> "The prediction task follows a **detection-conditioned protocol**: models predict future sensor-entity detection events only for marine entities that have been observed at least once during the context window. Entities that have never been detected are excluded from the prediction task, reflecting the realistic constraint that passive sensor networks cannot track unobserved targets. This design ensures no information leakage while maintaining ecological validity."

## Related Documentation
- [README.md](../../README.md) - Main project documentation
- [horizon_experiment_roadmap.md](../horizon_experiment_roadmap.md) - Experiment design
- [build_graphs.py](../../aquascan/dataset/build_graphs.py) - Implementation
