# 🧠 CELL 6A: GNN Training Only (Run Once)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🎮 Training on: {device}")

# IMPORTANT: Set PYTHONPATH for imports
import os
os.environ['PYTHONPATH'] = '/content/aquascan-gnns'

print("🚀 Training GNN models for all horizons...")
print("⚠️  This will take a while - run once, then use Cell 6B for evaluation tweaks")

for horizon in horizons:
    print(f"\n{'='*60}")
    print(f"🧠 Training GNN for {horizon}-tick horizon...")
    print(f"{'='*60}")

    # Create directories
    !mkdir -p /content/aquascan-gnns/results
    !mkdir -p /content/aquascan-gnns/checkpoints

    # Check if model already exists
    checkpoint_path = f'/content/aquascan-gnns/checkpoints/gnn_{horizon}tick.pt'
    if os.path.exists(checkpoint_path):
        print(f"✅ Model already exists: {checkpoint_path}")
        print(f"   Skipping training for {horizon}-tick horizon")
        continue

    print(f"🔄 Training new model for {horizon}-tick horizon...")
    
    # Train the model
    !cd /content/aquascan-gnns && python -m scripts.gnn_train \
        --data data/processed_{horizon}tick \
        --epochs 100 \
        --batch-size 32 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --patience 5

    # Rename checkpoint to horizon-specific name
    if os.path.exists('/content/aquascan-gnns/checkpoints/best.pt'):
        !cd /content/aquascan-gnns && mv checkpoints/best.pt checkpoints/gnn_{horizon}tick.pt
        print(f"✅ Model saved: checkpoints/gnn_{horizon}tick.pt")
    else:
        print(f"❌ Training failed for {horizon}-tick horizon!")

print(f"\n{'='*60}")
print("🎉 TRAINING COMPLETE!")
print(f"{'='*60}")

# Check what models we have
print("\n📁 Trained models:")
!ls -la /content/aquascan-gnns/checkpoints/gnn_*tick.pt

print("\n💡 Next step: Run Cell 6B to evaluate these models")
