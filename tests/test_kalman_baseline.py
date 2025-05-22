#!/usr/bin/env python3
"""
Test for Kalman baseline evaluation.

This test loads three graphs and checks that the JSON results have the expected keys.
"""
import json
import pytest
import torch
import os
from pathlib import Path


def test_kalman_baseline_results():
    """Test that kalman_baseline.json exists and has expected keys."""
    results_path = Path("results/kalman_baseline.json")
    
    # Check that results file exists
    assert results_path.exists(), "results/kalman_baseline.json not found"
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Check that all expected keys are present
    expected_keys = ["AUC", "Precision", "Recall"]
    for key in expected_keys:
        assert key in results, f"Missing key: {key}"
    
    # Check that values are reasonable
    assert 0 <= results["AUC"] <= 1, "AUC should be between 0 and 1"
    assert 0 <= results["Precision"] <= 1, "Precision should be between 0 and 1"
    assert 0 <= results["Recall"] <= 1, "Recall should be between 0 and 1"
    
    print(f"✓ Results file contains expected keys: {expected_keys}")
    print(f"✓ AUC: {results['AUC']:.4f}")
    print(f"✓ Precision: {results['Precision']:.4f}")
    print(f"✓ Recall: {results['Recall']:.4f}")


def test_data_loading():
    """Test that we can load three graphs from test data."""
    test_data_path = Path("data/processed/test.pt")
    
    # Check test data exists
    assert test_data_path.exists(), "test.pt not found"
    
    # Load test data
    graphs = torch.load(test_data_path, weights_only=False)
    
    # Check we have at least 3 graphs
    assert len(graphs) >= 3, f"Expected at least 3 graphs, got {len(graphs)}"
    
    # Check first three graphs have required structure
    for i in range(3):
        graph = graphs[i]
        
        # Check node types
        assert 'epsilon' in graph.node_types, f"Graph {i} missing epsilon nodes"
        assert 'theta' in graph.node_types, f"Graph {i} missing theta nodes"
        
        # Check edge types
        expected_edge_types = [
            ('epsilon', 'communicates', 'epsilon'),
            ('epsilon', 'detects', 'theta'),
            ('epsilon', 'will_detect', 'theta')
        ]
        
        for edge_type in expected_edge_types:
            assert edge_type in graph.edge_types, f"Graph {i} missing edge type: {edge_type}"
        
        # Check global attributes
        assert hasattr(graph, 'context_len'), f"Graph {i} missing context_len"
        assert hasattr(graph, 'horizon_len'), f"Graph {i} missing horizon_len"
    
    print(f"✓ Successfully loaded {len(graphs)} graphs")
    print(f"✓ First 3 graphs have expected structure")


if __name__ == "__main__":
    # Run tests directly
    test_data_loading()
    test_kalman_baseline_results()
    print("All tests passed!")
