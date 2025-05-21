"""
Test the HDF5 snapshot writer.

This ensures that the Aquascan simulation can correctly export state
to HDF5 files for further analysis and model training.
"""

import pytest
from pathlib import Path
import numpy as np
import json
from aquascan.run_simulation import run

def test_h5_created(tmp_path: Path):
    """
    Test that an HDF5 snapshot file is correctly created and populated.
    
    Uses pytest's tmp_path fixture to ensure the test is self-contained
    and doesn't create files in the project directory.
    """
    # Create a temporary output path
    out_file = tmp_path / "run.h5"
    
    # Run a short simulation with HDF5 export
    run(ticks=3, seed=1, visual=False, out_path=out_file)
    
    # Verify the file was created
    assert out_file.exists(), f"HDF5 file was not created at {out_file}"
    
    # Import h5py here to not require it for all tests
    import h5py
    
    # Check the file structure and content
    with h5py.File(out_file, "r") as f:
        # Check datasets exist
        assert "nodes" in f, "nodes dataset not found in HDF5 file"
        assert "edges" in f, "edges dataset not found in HDF5 file"
        assert "globals" in f, "globals dataset not found in HDF5 file"
        
        # Check datasets have the expected shape
        assert f["nodes"].shape[0] > 0, "nodes dataset is empty"
        
        # We expect 570 epsilon nodes * 3 ticks + some theta contacts
        expected_min_nodes = 570 * 3
        assert f["nodes"].shape[0] >= expected_min_nodes, f"Expected at least {expected_min_nodes} nodes"
        
        # Check data types
        assert f["nodes"].dtype.names == ("t", "gid", "type", "x", "y", "feat"), "Incorrect node dtype"
        assert f["edges"].dtype.names == ("t", "src", "dst", "rel"), "Incorrect edge dtype"
        
        # Check metadata 
        meta = json.loads(f["globals"][()])
        assert "seed" in meta, "seed not found in metadata"
        assert meta["seed"] == 1, f"Expected seed=1, got {meta['seed']}"
        assert meta["ticks"] == 3, f"Expected ticks=3, got {meta['ticks']}"
        
        # Basic integrity checks
        node_data = f["nodes"][:]
        
        # All ticks should be 0, 1, or 2
        tick_values = np.unique(node_data["t"])
        assert set(tick_values) == {0, 1, 2}, f"Expected ticks 0,1,2, got {tick_values}"
        
        # Check we have epsilon and theta nodes
        node_types = np.unique(node_data["type"])
        assert set(node_types) == {0, 1}, f"Expected node types 0,1, got {node_types}"
        
        print(f"Verified HDF5 file with {f['nodes'].shape[0]} nodes and {f['edges'].shape[0]} edges")
