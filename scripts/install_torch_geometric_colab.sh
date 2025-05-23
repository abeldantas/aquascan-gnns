#!/bin/bash
# Quick fix for torch_geometric installation in Colab

echo "🔧 Installing PyTorch Geometric with dependencies..."

# Install base requirements first
pip install -q -r requirements.txt
pip install -q tqdm joblib

# Detect PyTorch and CUDA versions
python3 << 'EOF'
import torch
import os

print(f"\n📊 System Info:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    cuda_version = torch.version.cuda.replace('.', '')
    torch_version = torch.__version__.split('+')[0]
    print(f"   CUDA version: {torch.version.cuda}")
    
    # Create install command
    cuda_tag = f"cu{cuda_version[:3]}"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    
    with open('/tmp/install_pyg.sh', 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv -f {wheel_url}\n")
        f.write(f"pip install -q torch-geometric\n")
else:
    # CPU version
    with open('/tmp/install_pyg.sh', 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv\n")
        f.write(f"pip install -q torch-geometric\n")
EOF

# Run the generated install script
chmod +x /tmp/install_pyg.sh
/tmp/install_pyg.sh

# Verify installation
python3 -c "import torch_geometric; print(f'\n✅ PyTorch Geometric {torch_geometric.__version__} installed successfully!')" || echo "❌ Installation failed!"
