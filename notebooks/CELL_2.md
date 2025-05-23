
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