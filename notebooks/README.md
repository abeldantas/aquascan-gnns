# 🚀 Aquascan Colab Pipeline - Quick Start Guide

## Step 1: Get Your GitHub Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate a token with `repo` permissions
3. Copy the token (you'll paste it in Colab)

## Step 2: Upload the Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Upload notebook
3. Upload `aquascan_full_pipeline.ipynb` from this folder

## Step 3: Enable GPU (Important!)

1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 or better)
3. Save

## Step 4: Run the Pipeline

The notebook will:
1. Clone your repo using the GitHub token
2. Generate simulation data (2-3 hours)
3. Build graph datasets (45-60 min)
4. Train GNN models on GPU (30-45 min)
5. Compare with baselines
6. Push results back to GitHub

## Tips for Colab Pro

- **Keep tab active**: Prevents disconnection
- **Check resources**: First cells show GPU/CPU info
- **Save frequently**: Results go to Google Drive
- **Monitor progress**: Each section has progress bars

## Customization

In the notebook, you can adjust:
- `NUM_RUNS`: Number of simulations (10 for test, 1000 for full)
- `horizons`: Which prediction horizons to test
- `device`: Force CPU if needed (but why would you?)

## After Running

Results will be in:
- **GitHub**: `results/` folder with all metrics
- **Google Drive**: Timestamped archives of everything
- **Checkpoints**: Trained models ready to use

## Emergency Recovery

If Colab disconnects:
1. Re-run setup cells
2. Skip data generation if `data/raw_5tick/` exists
3. Continue from where it stopped

The notebook is designed to be re-entrant!
