"""
Test the graph builder functionality.

This module tests the windowed graph builder to ensure it correctly converts
raw HDF5 files into fixed-length spatiotemporal graphs for machine learning.
"""

import os
import glob
import random
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import h5py
import numpy as np

from aquascan.dataset.build_graphs import h5_to_graphs, save_split, validate_graphs


@pytest.fixture
def temp_raw_dir():
    """Create a temporary directory for raw HDF5 files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_out_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_h5_to_graphs():
    """Test converting HDF5 files to graphs with small windows."""
    # Find existing HDF5 files (generated from batch process)
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        pytest.skip("No raw HDF5 files found, skipping test")
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.h5"))
    if not raw_files:
        pytest.skip("No raw HDF5 files found, skipping test")
    
    # Use a small sample of files for testing
    sample_file = random.choice(raw_files)
    
    # Use small window sizes for faster testing
    context_len = 5
    horizon_len = 1
    
    # Convert to graphs
    graphs = h5_to_graphs(sample_file, context_len, horizon_len)
    
    # Basic validation
    assert len(graphs) > 0, "Failed to create any graphs"
    
    # Check graph structure
    for graph in graphs[:5]:  # Check the first few graphs
        # Check node types
        assert 'epsilon' in graph.node_types, "Missing epsilon nodes"
        assert 'theta' in graph.node_types, "Missing theta nodes"
        
        # Check edge types
        assert ('epsilon', 'communicates', 'epsilon') in graph.edge_types, "Missing communication edges"
        assert ('epsilon', 'detects', 'theta') in graph.edge_types, "Missing detection edges"
        assert ('epsilon', 'will_detect', 'theta') in graph.edge_types, "Missing target edges"
        
        # Check that context and horizon lengths are set
        assert hasattr(graph, 'context_len'), "Missing context_len attribute"
        assert hasattr(graph, 'horizon_len'), "Missing horizon_len attribute"
        assert graph.context_len == context_len, "Incorrect context_len"
        assert graph.horizon_len == horizon_len, "Incorrect horizon_len"
        
        # Check node features
        if graph['epsilon'].x.shape[0] > 0:
            assert graph['epsilon'].x.shape[1] == 4, "Epsilon nodes should have 4 features"
        
        if graph['theta'].x.shape[0] > 0:
            assert graph['theta'].x.shape[1] == 4, "Theta nodes should have 4 features"
        
        # Check that target labels have the right shape
        target_edge_type = ('epsilon', 'will_detect', 'theta')
        if hasattr(graph[target_edge_type], 'edge_label') and hasattr(graph[target_edge_type], 'edge_label_index'):
            assert graph[target_edge_type].edge_label.shape[0] == graph[target_edge_type].edge_label_index.shape[1], \
                "Target label shape mismatch"


def test_graph_split(temp_out_dir):
    """Test splitting and saving graphs."""
    # Find existing HDF5 files
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        pytest.skip("No raw HDF5 files found, skipping test")
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.h5"))
    if not raw_files:
        pytest.skip("No raw HDF5 files found, skipping test")
    
    # Use a small sample of files for testing
    sample_file = random.choice(raw_files)
    
    # Use small window sizes for faster testing
    context_len = 5
    horizon_len = 1
    
    # Convert to graphs
    graphs = h5_to_graphs(sample_file, context_len, horizon_len)
    
    # Limit to a small number of graphs for testing
    graphs = graphs[:30]
    
    # Split ratios
    split_ratios = [0.7, 0.15, 0.15]
    
    # Split and save
    save_split(graphs, split_ratios, temp_out_dir, adv_fraction=0.1)
    
    # Check that files were created
    assert os.path.exists(os.path.join(temp_out_dir, 'train.pt')), "Missing train.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'val.pt')), "Missing val.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'test.pt')), "Missing test.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'adversarial.pt')), "Missing adversarial.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'meta.json')), "Missing meta.json"
    
    # Try loading the files to ensure they're valid
    train_graphs = torch.load(os.path.join(temp_out_dir, 'train.pt'))
    val_graphs = torch.load(os.path.join(temp_out_dir, 'val.pt'))
    test_graphs = torch.load(os.path.join(temp_out_dir, 'test.pt'))
    adv_graphs = torch.load(os.path.join(temp_out_dir, 'adversarial.pt'))
    
    # Check that the splits have the expected sizes
    assert len(train_graphs) + len(val_graphs) + len(test_graphs) == len(graphs), "Split sizes don't add up"
    assert len(adv_graphs) > 0, "No adversarial graphs were created"


def test_validate_graphs():
    """Test the graph validation function."""
    # Find existing HDF5 files
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        pytest.skip("No raw HDF5 files found, skipping test")
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.h5"))
    if not raw_files:
        pytest.skip("No raw HDF5 files found, skipping test")
    
    # Use a small sample of files for testing
    sample_file = random.choice(raw_files)
    
    # Use small window sizes for faster testing
    context_len = 5
    horizon_len = 1
    
    # Convert to graphs
    graphs = h5_to_graphs(sample_file, context_len, horizon_len)
    
    # Limit to a small number of graphs for testing
    graphs = graphs[:10]
    
    # Validate
    assert validate_graphs(graphs), "Graph validation failed"


def test_command_line(temp_out_dir):
    """Test the command line interface."""
    # Find existing HDF5 files
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        pytest.skip("No raw HDF5 files found, skipping test")
    
    raw_files = glob.glob(os.path.join(raw_dir, "*.h5"))
    if not raw_files:
        pytest.skip("No raw HDF5 files found, skipping test")
    
    # Use a small subset for testing
    import subprocess
    
    # Run the command with small window sizes and limit=1
    cmd = [
        "python", "-m", "aquascan.dataset.build_graphs",
        "--raw", raw_dir,
        "--out", temp_out_dir,
        "--context", "5",
        "--horizon", "1",
        "--split", "0.7", "0.15", "0.15",
        "--adv_fraction", "0.1",
        "--limit", "1"  # Process only one file for speed
    ]
    
    # Execute the command
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    
    # Check return code
    assert result.returncode == 0, f"Command failed with: {result.stderr}"
    
    # Check that files were created
    assert os.path.exists(os.path.join(temp_out_dir, 'train.pt')), "Missing train.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'val.pt')), "Missing val.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'test.pt')), "Missing test.pt"
    assert os.path.exists(os.path.join(temp_out_dir, 'meta.json')), "Missing meta.json"
