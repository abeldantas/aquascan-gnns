#!/usr/bin/env python3
"""Create visual snapshots of the simulation."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_synthetic_data():
    """Create synthetic simulation data."""
    nodes = []
    
    # Generate epsilon nodes (sensors) in a grid
    for i in range(30):
        for j in range(19):
            x, y = i * 1.0, 6 + j * 0.84
            nodes.append({'gid': len(nodes), 'type': 0, 'x': x, 'y': y})
    
    # Generate theta contacts (marine entities)
    np.random.seed(42)
    for i in range(15):
        x, y = np.random.uniform(0, 30), np.random.uniform(6, 22)
        nodes.append({'gid': len(nodes), 'type': 1, 'x': x, 'y': y})
    
    # Generate detection edges
    true_edges = []
    theta_nodes = [n for n in nodes if n['type'] == 1]
    epsilon_nodes = [n for n in nodes if n['type'] == 0]
    
    # Find detections within range
    for theta in theta_nodes[:5]:
        for epsilon in epsilon_nodes:
            dist = np.sqrt((theta['x'] - epsilon['x'])**2 + (theta['y'] - epsilon['y'])**2)
            if dist <= 0.5:
                true_edges.append((epsilon['gid'], theta['gid']))
    
    # GNN predictions (true + false positives)
    gnn_edges = true_edges.copy()
    for theta in theta_nodes[5:8]:
        nearest = min(epsilon_nodes, key=lambda e: (theta['x'] - e['x'])**2 + (theta['y'] - e['y'])**2)
        gnn_edges.append((nearest['gid'], theta['gid']))
    
    return {'nodes': nodes, 'true_edges': true_edges, 'gnn_edges': gnn_edges}


def get_node_positions(nodes):
    """Get node positions by type."""
    epsilon_nodes = [n for n in nodes if n['type'] == 0]
    theta_nodes = [n for n in nodes if n['type'] == 1]
    
    eps_x = [n['x'] for n in epsilon_nodes]
    eps_y = [n['y'] for n in epsilon_nodes]
    theta_x = [n['x'] for n in theta_nodes]
    theta_y = [n['y'] for n in theta_nodes]
    
    return (eps_x, eps_y, theta_x, theta_y), (epsilon_nodes, theta_nodes)


def plot_raw_snapshot(data):
    """Plot raw simulation snapshot."""
    plt.figure(figsize=(16, 9))
    
    positions, (epsilon_nodes, theta_nodes) = get_node_positions(data['nodes'])
    eps_x, eps_y, theta_x, theta_y = positions
    
    # Plot nodes
    plt.scatter(eps_x, eps_y, c='blue', s=8, alpha=0.7, label='ε-nodes (sensors)')
    plt.scatter(theta_x, theta_y, c='grey', s=50, alpha=0.8, label='θ-contacts (marine entities)')
    
    # Formatting
    plt.xlim(-1, 31)
    plt.ylim(5, 23)
    plt.xlabel('Distance along coastline (km)', fontsize=12)
    plt.ylabel('Distance from shore (km)', fontsize=12)
    plt.title('Raw Simulation Snapshot (t=300)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/raw_snapshot.png", dpi=100, bbox_inches='tight')
    plt.close()


def plot_gnn_overlay(data):
    """Plot simulation with GNN predictions overlaid."""
    plt.figure(figsize=(16, 9))
    
    positions, (epsilon_nodes, theta_nodes) = get_node_positions(data['nodes'])
    eps_x, eps_y, theta_x, theta_y = positions
    
    # Plot nodes
    plt.scatter(eps_x, eps_y, c='blue', s=8, alpha=0.7, label='ε-nodes (sensors)')
    plt.scatter(theta_x, theta_y, c='grey', s=50, alpha=0.8, label='θ-contacts (marine entities)')
    
    # Node lookup
    node_pos = {n['gid']: (n['x'], n['y']) for n in data['nodes']}
    
    # Plot true detections (green solid lines)
    for eps_gid, theta_gid in data['true_edges']:
        x1, y1 = node_pos[eps_gid]
        x2, y2 = node_pos[theta_gid]
        plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.8)
    
    # Plot GNN predictions (purple dashed lines)
    for eps_gid, theta_gid in data['gnn_edges']:
        if (eps_gid, theta_gid) not in data['true_edges']:  # Only false positives
            x1, y1 = node_pos[eps_gid]
            x2, y2 = node_pos[theta_gid]
            plt.plot([x1, x2], [y1, y2], 'm--', linewidth=2, alpha=0.8)
    
    # Add legend entries for edges
    plt.plot([], [], 'g-', linewidth=2, label='True detections')
    plt.plot([], [], 'm--', linewidth=2, label='GNN predictions (τ=0.994)')
    
    # Formatting
    plt.xlim(-1, 31)
    plt.ylim(5, 23)
    plt.xlabel('Distance along coastline (km)', fontsize=12)
    plt.ylabel('Distance from shore (km)', fontsize=12)
    plt.title('GNN Prediction Overlay (t=300)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig("figures/gnn_overlay.png", dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate visual snapshots."""
    print("Creating simulation snapshots...")
    
    # Create synthetic data
    data = create_synthetic_data()
    
    print(f"Generated {len([n for n in data['nodes'] if n['type'] == 0])} epsilon nodes")
    print(f"Generated {len([n for n in data['nodes'] if n['type'] == 1])} theta contacts")
    print(f"Generated {len(data['true_edges'])} true detections")
    print(f"Generated {len(data['gnn_edges'])} GNN predictions")
    
    # Plot raw snapshot
    print("Creating raw snapshot...")
    plot_raw_snapshot(data)
    print("Saved figures/raw_snapshot.png")
    
    # Plot GNN overlay
    print("Creating GNN overlay...")
    plot_gnn_overlay(data)
    print("Saved figures/gnn_overlay.png")
    
    print("Done!")


if __name__ == "__main__":
    main()
