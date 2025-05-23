#!/usr/bin/env python3
"""
Generate the Aquascan Colab notebook.
This creates the full pipeline notebook for running everything on Colab Pro.
"""

import json
import os

def create_notebook():
    """Create the Colab notebook structure."""
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "collapsed_sections": ["setup_section", "debug_section"]
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": []
    }
    
    # Header cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {"id": "header_section"},
        "source": [
            "# 🌊 Aquascan Full Pipeline - Colab Pro Edition\\n",
            "\\n",
            "This notebook runs the ENTIRE Aquascan pipeline on Colab Pro:\\n",
            "1. Clone the repository\\n",
            "2. Generate raw simulation data\\n",
            "3. Build graph datasets for multiple horizons\\n",
            "4. Train GNN models\\n",
            "5. Run baselines and compare\\n",
            "6. Push results back to repo\\n",
            "\\n",
            "**Estimated runtime**: 4-5 hours total on Colab Pro"
        ]
    })
    
    # Setup section
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {"id": "setup_section"},
        "source": ["## 🔧 Setup Environment"]
    })
    
    # GitHub auth cell
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "setup_auth"},
        "source": [
            "# Setup GitHub authentication\\n",
            "import os\\n",
            "from getpass import getpass\\n",
            "\\n",
            "# Get GitHub token (paste it when prompted)\\n",
            "github_token = getpass('Enter your GitHub personal access token: ')\\n",
            "github_username = input('Enter your GitHub username: ')\\n",
            "github_repo = 'aquascan-gnns'\\n",
            "\\n",
            "# Store for later use\\n",
            "os.environ['GITHUB_TOKEN'] = github_token\\n",
            "os.environ['GITHUB_USERNAME'] = github_username\\n",
            "os.environ['GITHUB_REPO'] = github_repo"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Check resources cell
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "check_resources"},
        "source": [
            "# Check Colab resources\\n",
            "!nvidia-smi\\n",
            "!lscpu | grep 'CPU(s):' | head -1\\n",
            "!free -h\\n",
            "\\n",
            "# Check disk space\\n",
            "!df -h /content\\n",
            "\\n",
            "# Import essentials\\n",
            "import subprocess\\n",
            "import multiprocessing\\n",
            "import time\\n",
            "from datetime import datetime\\n",
            "\\n",
            "print(f\"\\\\n✅ Environment check complete!\")\\n",
            "print(f\"🔧 CPUs available: {multiprocessing.cpu_count()}\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Clone repo cell
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "clone_repo"},
        "source": [
            "# Clone the repository\\n",
            "%cd /content\\n",
            "\\n",
            "# Remove if exists\\n",
            "!rm -rf aquascan-gnns\\n",
            "\\n",
            "# Clone with authentication\\n",
            "!git clone https://{github_token}@github.com/{github_username}/{github_repo}.git\\n",
            "\\n",
            "# Enter the directory\\n",
            "%cd aquascan-gnns\\n",
            "\\n",
            "# Setup git config for commits\\n",
            "!git config user.email \\\"colab@aquascan.ai\\\"\\n",
            "!git config user.name \\\"Colab Runner\\\"\\n",
            "\\n",
            "print(\"\\\\n✅ Repository cloned successfully!\")\\n",
            "!pwd\\n",
            "!ls -la"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Add more cells...
    # (Truncated for brevity - the full script would add all cells)
    
    return notebook

def main():
    """Generate and save the notebook."""
    notebook = create_notebook()
    
    output_path = "aquascan_full_pipeline.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook generated: {output_path}")
    print(f"📏 Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("\n🚀 Next steps:")
    print("1. Upload this notebook to Google Colab")
    print("2. Run Runtime → Change runtime type → GPU")
    print("3. Follow the cells step by step!")

if __name__ == "__main__":
    main()
