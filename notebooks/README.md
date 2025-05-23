# 🚀 Aquascan Colab Pipeline - Quick Start Guide

## ⚠️ Important: Updated Pipeline (v2.0)

This pipeline has been battle-tested and includes fixes for common Colab issues:
- ✅ PyTorch Geometric installation
- ✅ Shared memory errors  
- ✅ Correct script arguments
- ✅ Memory management
- ✅ PYTHONPATH issues

## Step 1: Get Your GitHub Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate a token with `repo` permissions
3. Copy the token (you'll paste it in Colab)

## Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → GPU (T4 or better)**
4. Save

## Step 3: Copy the Pipeline Code

Open [COLAB_PIPELINE.md](COLAB_PIPELINE.md) and copy each cell into your Colab notebook.

**Critical cells that have been fixed:**
- Cell 2: PyTorch Geometric installation
- Cell 5: Shared memory fix for multiprocessing
- Cell 6: Correct argument names for scripts
- Cell 7: Kalman baseline wrapper

## Step 4: Run the Pipeline!

1. **Run Cell 1** - Enter your GitHub credentials
2. **Run remaining cells** in order (~4-5 hours total)

## Common Issues & Solutions

### PyTorch Geometric Won't Install
Already fixed in Cell 2, but if issues persist, the cell includes fallback methods.

### Shared Memory Error
Already fixed in Cell 5 with:
```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

### Module Not Found
Already fixed with PYTHONPATH in Cells 6 & 7.

## What You'll Get

After ~4-5 hours:
- ✅ Results pushed to your GitHub repo
- ✅ Performance plots in `results/`
- ✅ Trained models in `checkpoints/`
- ✅ Backups in Google Drive

## Pro Tips

1. **Use Colab Pro** - More RAM, better GPUs, longer sessions
2. **Keep tab active** - Prevents disconnection
3. **Monitor Cell 5** - Graph building shows progress bars
4. **Check Cell 11** - Summary shows if hypothesis confirmed

## Emergency Recovery

If Colab disconnects:
1. Re-run Cells 1-2 (setup)
2. Check what exists: `!ls data/`
3. Skip completed steps
4. Continue from where it stopped

The pipeline is designed to be resumable!
