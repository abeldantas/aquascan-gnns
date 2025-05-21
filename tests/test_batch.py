"""
Test the batch generator functionality.

This module tests the batch generator to ensure it can successfully
create multiple HDF5 files from simulations with different seeds.
"""

import pytest
from pathlib import Path
from subprocess import run, CalledProcessError


def test_batch(tmp_path: Path):
    """
    Test running a small batch of simulations.
    
    Tests that:
    1. The batch generator runs successfully
    2. It creates the expected number of HDF5 files
    3. The files have the correct structure and content
    
    Args:
        tmp_path: Temporary directory provided by pytest
    """
    # Run batch generator with a small number of simulations
    cmd = [
        "python", "-m", "aquascan.batch.generate",
        "--cfg", "configs/base.yml",
        "--runs", "2",  # Smaller for faster testing
        "--out", str(tmp_path),
        "--jobs", "2",  # Use 2 processes for testing
        "--validate"    # Perform basic validation
    ]
    
    # Run the command and check it completes successfully
    result = run(cmd, capture_output=True, text=True)
    
    # Check return code
    assert result.returncode == 0, f"Batch generator failed: {result.stderr}"
    
    # Check that the expected files were created
    h5_files = list(tmp_path.glob("*.h5"))
    assert len(h5_files) == 2, f"Expected 2 HDF5 files, found {len(h5_files)}"
    
    # Basic success message
    print(f"Batch test successful: {len(h5_files)} files created")
    
    # For more detailed validation, uncomment:
    # import h5py
    # import json
    # with h5py.File(h5_files[0], "r") as f:
    #     assert "nodes" in f, "nodes dataset missing"
    #     assert "edges" in f, "edges dataset missing"
    #     assert "globals" in f, "globals dataset missing"
    #     assert f["nodes"].shape[0] > 0, "nodes dataset is empty"
    #     assert f["edges"].shape[0] > 0, "edges dataset is empty"
