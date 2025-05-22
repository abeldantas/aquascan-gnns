#!/usr/bin/env python3
"""Test evaluation scripts for correctness and performance."""

import json
import subprocess
import sys
from pathlib import Path


def test_make_metrics_table():
    """Test that make_metrics_table.py works correctly."""
    print("Testing make_metrics_table.py...")
    
    result = subprocess.run([
        sys.executable, "scripts/make_metrics_table.py"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # Check that output file was created
    assert Path("results/metrics_table.json").exists(), "Output file not created"
    
    # Verify output structure
    with open("results/metrics_table.json", 'r') as f:
        output = json.load(f)
    
    assert "Kalman" in output, "Missing Kalman results in output"
    assert "GNN" in output, "Missing GNN results in output"
    assert "Gain" in output, "Missing Gain calculation in output"
    
    print("✅ make_metrics_table.py test passed")


def test_plot_curves():
    """Test that plot_curves.py works correctly."""
    print("Testing plot_curves.py...")
    
    result = subprocess.run([
        sys.executable, "scripts/plot_curves.py"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    # Check that figure files were created
    assert Path("figures/roc_curve.png").exists(), "ROC curve not created"
    assert Path("figures/pr_curve.png").exists(), "PR curve not created"
    
    # Check file sizes (should be reasonable, not empty)
    roc_size = Path("figures/roc_curve.png").stat().st_size
    pr_size = Path("figures/pr_curve.png").stat().st_size
    
    assert roc_size > 1000, f"ROC curve file too small: {roc_size} bytes"
    assert pr_size > 1000, f"PR curve file too small: {pr_size} bytes"
    
    # Check that files are under 300KB as required
    assert roc_size < 300_000, f"ROC curve file too large: {roc_size} bytes"
    assert pr_size < 300_000, f"PR curve file too large: {pr_size} bytes"
    
    print("✅ plot_curves.py test passed")


def test_make_snapshots():
    """Test that make_snapshots.py works correctly."""
    print("Testing make_snapshots.py...")
    
    result = subprocess.run([
        sys.executable, "scripts/make_snapshots.py"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"


    # Check that snapshot files were created
    assert Path("figures/raw_snapshot.png").exists(), "Raw snapshot not created"
    assert Path("figures/gnn_overlay.png").exists(), "GNN overlay not created"
    
    # Check file sizes
    raw_size = Path("figures/raw_snapshot.png").stat().st_size
    overlay_size = Path("figures/gnn_overlay.png").stat().st_size
    
    assert raw_size > 1000, f"Raw snapshot file too small: {raw_size} bytes"
    assert overlay_size > 1000, f"GNN overlay file too small: {overlay_size} bytes"
    
    # Check that files are under 300KB as required
    assert raw_size < 300_000, f"Raw snapshot file too large: {raw_size} bytes"
    assert overlay_size < 300_000, f"GNN overlay file too large: {overlay_size} bytes"
    
    print("✅ make_snapshots.py test passed")


def main():
    """Run all tests."""
    print("Running evaluation script tests...")
    
    # Test all scripts
    test_make_metrics_table()
    test_plot_curves()
    test_make_snapshots()
    
    print("\n🎉 All tests passed!")
    print("Evaluation package is ready for publication.")


if __name__ == "__main__":
    main()
