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