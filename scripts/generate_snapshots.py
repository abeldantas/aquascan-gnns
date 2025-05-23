#!/usr/bin/env python3
"""
Generate visual snapshots from headless simulation at various time points.

This script runs the Aquascan simulation in headless mode and generates
matplotlib visualizations at specified intervals to verify the simulation
is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aquascan.simulation.simulation_loop import AquascanSimulation
from aquascan.config.simulation_config import (
    AREA_LENGTH, AREA_WIDTH, SHORE_DISTANCE, DETECTION_RADIUS
)


def plot_simulation_state(sim, tick, output_path, show_labels=True, trajectory_history=None, interval=5):
    """
    Create a matplotlib visualization of the simulation state with trajectory lines.
    
    Args:
        sim: AquascanSimulation instance
        tick: Current tick number
        output_path: Path to save the image
        show_labels: Whether to show node/entity labels
        trajectory_history: Dictionary storing position history for each entity
        interval: Tick interval for trajectory recording
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set plot boundaries with some margin
    margin = 3
    ax.set_xlim(-margin, AREA_LENGTH + margin)
    ax.set_ylim(SHORE_DISTANCE - margin, SHORE_DISTANCE + AREA_WIDTH + margin)
    
    # Add deployment area rectangle
    deployment_rect = patches.Rectangle(
        (0, SHORE_DISTANCE), AREA_LENGTH, AREA_WIDTH,
        linewidth=2, edgecolor='gray', facecolor='none',
        linestyle='--', alpha=0.5
    )
    ax.add_patch(deployment_rect)
    
    # Plot epsilon node connections
    # Collect permanent connections
    permanent_lines = []
    for (id1, id2) in sim.epsilon_connections['permanent']:
        # Find nodes by ID
        node1 = next((n for n in sim.epsilon_nodes if n.id == id1), None)
        node2 = next((n for n in sim.epsilon_nodes if n.id == id2), None)
        if node1 and node2:
            permanent_lines.append([node1.position, node2.position])
    
    # Plot permanent connections
    if permanent_lines:
        lc_permanent = LineCollection(permanent_lines, colors='blue', 
                                    linewidths=0.5, alpha=0.5)
        ax.add_collection(lc_permanent)
    
    # Collect intermittent connections
    intermittent_lines = []
    for (id1, id2) in sim.epsilon_connections['intermittent']:
        node1 = next((n for n in sim.epsilon_nodes if n.id == id1), None)
        node2 = next((n for n in sim.epsilon_nodes if n.id == id2), None)
        if node1 and node2:
            intermittent_lines.append([node1.position, node2.position])
    
    # Plot intermittent connections
    if intermittent_lines:
        lc_intermittent = LineCollection(intermittent_lines, colors='lightblue',
                                       linewidths=0.5, alpha=0.3, linestyles='dashed')
        ax.add_collection(lc_intermittent)
    
    # Plot epsilon nodes
    epsilon_x = [node.position[0] for node in sim.epsilon_nodes]
    epsilon_y = [node.position[1] for node in sim.epsilon_nodes]
    ax.scatter(epsilon_x, epsilon_y, c='blue', s=20, alpha=0.7, 
              marker='o', label=f'ε-nodes ({len(sim.epsilon_nodes)})')
    
    # Generate unique colors for each contact
    contact_colors = {}
    cmap = cm.get_cmap('tab10')  # Use a colormap with distinct colors
    for i, contact in enumerate(sim.theta_contacts):
        # Use modulo to cycle through colors if we have more than 10 entities
        contact_colors[contact.id] = cmap(i % 10)
    
    # Plot trajectories if history is available
    if trajectory_history and tick > 0:
        for contact_id, positions in trajectory_history.items():
            if len(positions) > 1:
                # Create line segments for trajectory
                trajectory_points = np.array(positions)
                # Plot trajectory line with entity-specific color
                ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
                       color=contact_colors.get(contact_id, 'gray'), 
                       linewidth=2, alpha=0.6, linestyle='-')
                
                # Add small circles at each recorded position
                ax.scatter(trajectory_points[:-1, 0], trajectory_points[:-1, 1],
                          c=[contact_colors.get(contact_id, 'gray')], 
                          s=30, alpha=0.4, edgecolors='none')
    
    # Plot theta contacts and check detections
    detected_contacts = []
    undetected_contacts = []
    detection_lines = []
    
    for contact in sim.theta_contacts:
        is_detected = False
        for epsilon_node in sim.epsilon_nodes:
            distance = np.linalg.norm(epsilon_node.position - contact.position)
            if distance <= DETECTION_RADIUS:
                is_detected = True
                # Add detection line
                detection_lines.append([epsilon_node.position, contact.position])
        
        if is_detected:
            detected_contacts.append(contact)
        else:
            undetected_contacts.append(contact)
    
    # Plot detection connections
    if detection_lines:
        lc_detection = LineCollection(detection_lines, colors='purple',
                                    linewidths=1.5, alpha=0.8)
        ax.add_collection(lc_detection)
    
    # Plot detected contacts
    if detected_contacts:
        for contact in detected_contacts:
            color = contact_colors[contact.id]
            ax.scatter(contact.position[0], contact.position[1], 
                      c=[color], s=120, alpha=0.9,
                      marker='o', edgecolors='darkgreen', linewidths=3)
            
            if show_labels:
                ax.annotate(contact.id, (contact.position[0], contact.position[1]),
                          xytext=(0, -15), textcoords='offset points',
                          ha='center', fontsize=8, color=color, fontweight='bold')
    
    # Plot undetected contacts
    if undetected_contacts:
        for contact in undetected_contacts:
            color = contact_colors[contact.id]
            # Make undetected contacts more transparent
            ax.scatter(contact.position[0], contact.position[1], 
                      c=[color], s=120, alpha=0.4,
                      marker='o', edgecolors='gray', linewidths=1)
            
            if show_labels:
                ax.annotate(contact.id, (contact.position[0], contact.position[1]),
                          xytext=(0, -15), textcoords='offset points',
                          ha='center', fontsize=8, color=color, alpha=0.6)
    
    # Add ocean current indicator
    center_x, center_y = AREA_LENGTH / 2, SHORE_DISTANCE + AREA_WIDTH / 2
    current_vector = sim.ocean_area.calculate_ocean_current([center_x, center_y], sim.current_time)
    
    # Scale for visibility
    scale_factor = 10.0
    ax.arrow(AREA_LENGTH / 2, SHORE_DISTANCE - 2,
            current_vector[0] * scale_factor, current_vector[1] * scale_factor,
            head_width=0.8, head_length=0.5, fc='darkblue', ec='darkblue')
    
    # Calculate current info
    current_strength = np.sqrt(current_vector[0]**2 + current_vector[1]**2)
    current_direction = np.arctan2(current_vector[1], current_vector[0]) * 180 / np.pi
    
    # Add labels and title
    ax.set_xlabel('Distance along coastline (km)', fontsize=12)
    ax.set_ylabel('Distance from shore (km)', fontsize=12)
    
    # Calculate simulation time
    sim_hours = int(sim.current_time / 3600)
    sim_minutes = int((sim.current_time % 3600) / 60)
    sim_seconds = int(sim.current_time % 60)
    
    title = f'Aquascan Simulation - Tick {tick} (Time: {sim_hours:02d}:{sim_minutes:02d}:{sim_seconds:02d})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add statistics text
    stats_text = (
        f'Total Detections: {sim.stats["detections"]}\n'
        f'Current: {current_strength*100:.1f} cm/s @ {current_direction:.0f}°\n'
        f'Speed: x128\n'
        f'Detected: {len(detected_contacts)}/{len(sim.theta_contacts)}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend with custom entries
    legend_entries = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, alpha=0.7, label=f'ε-nodes ({len(sim.epsilon_nodes)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, markeredgecolor='darkgreen', markeredgewidth=2,
                   label=f'Detected θ-contacts ({len(detected_contacts)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=10, alpha=0.6, label=f'Undetected θ-contacts ({len(undetected_contacts)})'),
        plt.Line2D([0], [0], color='purple', linewidth=2, label='Active detections'),
    ]
    
    if trajectory_history and any(len(pos) > 1 for pos in trajectory_history.values()):
        legend_entries.append(plt.Line2D([0], [0], color='gray', linewidth=2, 
                                       linestyle='-', label='Entity trajectories'))
    
    ax.legend(handles=legend_entries, loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved snapshot: {output_path}")


def generate_snapshots(seed=42, total_ticks=600, snapshot_ticks=None, output_dir='visualizations', interval=5):
    """
    Generate visual snapshots from a headless simulation run with trajectory tracking.
    
    Args:
        seed: Random seed for reproducibility
        total_ticks: Total number of ticks to simulate
        snapshot_ticks: List of ticks to capture snapshots (default: evenly spaced)
        output_dir: Directory to save snapshots
        interval: Interval for snapshots and trajectory recording
    """
    # Default snapshot ticks if not provided
    if snapshot_ticks is None:
        # Capture at start, then every interval ticks
        snapshot_ticks = list(range(0, total_ticks + 1, interval))
        if total_ticks not in snapshot_ticks:
            snapshot_ticks.append(total_ticks)
    
    # Create output directory
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"run_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting headless simulation with seed={seed}")
    print(f"Will capture {len(snapshot_ticks)} snapshots over {total_ticks} ticks")
    print(f"Output directory: {run_dir}")
    
    # Initialize simulation
    sim = AquascanSimulation(seed=seed)
    sim.initialize()
    sim.start()
    
    # Set speed to x128
    sim.set_speed(128)
    
    # Initialize trajectory history - store positions at interval ticks only
    trajectory_history = defaultdict(list)
    
    # Run simulation and capture snapshots
    for tick in range(total_ticks + 1):
        # Record positions at interval ticks for trajectory
        if tick % interval == 0:
            for contact in sim.theta_contacts:
                trajectory_history[contact.id].append(contact.position.copy())
        
        # Capture snapshot if needed
        if tick in snapshot_ticks:
            output_file = run_dir / f"snapshot_tick_{tick:04d}.png"
            plot_simulation_state(sim, tick, output_file, 
                                show_labels=(tick == 0), 
                                trajectory_history=trajectory_history,
                                interval=interval)
        
        # Run simulation tick (unless we're at the end)
        if tick < total_ticks:
            sim.tick()
    
    sim.stop()
    
    print(f"\nSimulation complete! Snapshots saved to: {run_dir}")
    print(f"Statistics: {sim.stats}")
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visual snapshots from Aquascan simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ticks", type=int, default=600, help="Total simulation ticks")
    parser.add_argument("--interval", type=int, default=60, help="Snapshot interval in ticks")
    parser.add_argument("--output", type=str, default="visualizations", help="Output directory")
    
    args = parser.parse_args()
    
    # Generate snapshot ticks based on interval
    snapshot_ticks = list(range(0, args.ticks + 1, args.interval))
    
    generate_snapshots(
        seed=args.seed,
        total_ticks=args.ticks,
        snapshot_ticks=snapshot_ticks,
        output_dir=args.output,
        interval=args.interval
    )
