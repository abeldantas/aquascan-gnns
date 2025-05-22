#!/usr/bin/env python3
"""
Simple pytest-compatible tests for the evaluation package.
"""

import json
from pathlib import Path


def test_results_files_exist():
    """Test that all required result files exist."""
    assert Path("results/kalman_baseline.json").exists()
    assert Path("results/gnn_test.json").exists()
    assert Path("results/metrics_table.json").exists()


def test_figure_files_exist():
    """Test that all required figure files exist."""
    assert Path("figures/roc_curve.png").exists()
    assert Path("figures/pr_curve.png").exists()
    assert Path("figures/raw_snapshot.png").exists()
    assert Path("figures/gnn_overlay.png").exists()


def test_documentation_exists():
    """Test that documentation exists."""
    assert Path("docs/eval.md").exists()


def test_scripts_exist():
    """Test that all evaluation scripts exist."""
    assert Path("scripts/make_metrics_table.py").exists()
    assert Path("scripts/plot_curves.py").exists()
    assert Path("scripts/make_snapshots.py").exists()


def test_metrics_table_format():
    """Test that metrics table has correct format."""
    with open("results/metrics_table.json", 'r') as f:
        data = json.load(f)
    
    assert "Kalman" in data
    assert "GNN" in data
    assert "Gain" in data
    
    # Check that all models have required metrics
    for model in ["Kalman", "GNN"]:
        assert "AUC" in data[model]
        assert "Precision" in data[model]
        assert "Recall" in data[model]


def test_file_sizes():
    """Test that figure files are reasonable size."""
    figures = [
        "figures/roc_curve.png",
        "figures/pr_curve.png", 
        "figures/raw_snapshot.png",
        "figures/gnn_overlay.png"
    ]
    
    for fig_path in figures:
        file_size = Path(fig_path).stat().st_size
        assert file_size > 1000, f"{fig_path} too small: {file_size} bytes"
        assert file_size < 300_000, f"{fig_path} too large: {file_size} bytes"


if __name__ == "__main__":
    # Run tests manually if called directly
    test_results_files_exist()
    test_figure_files_exist()
    test_documentation_exists()
    test_scripts_exist()
    test_metrics_table_format()
    test_file_sizes()
    print("All pytest-compatible tests passed!")
