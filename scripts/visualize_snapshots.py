#!/usr/bin/env python3
"""
Create a grid visualization of multiple snapshots for easy comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import glob
from PIL import Image

def create_snapshot_grid(snapshot_dir, output_file='snapshot_grid.png', cols=3):
    """
    Create a grid of snapshots from a directory.
    
    Args:
        snapshot_dir: Directory containing snapshot PNG files
        output_file: Output filename for the grid
        cols: Number of columns in the grid
    """
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(str(Path(snapshot_dir) / "snapshot_tick_*.png")))
    
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_dir}")
        return
    
    print(f"Found {len(snapshot_files)} snapshots")
    
    # Load images
    images = [Image.open(f) for f in snapshot_files]
    
    # Calculate grid dimensions
    rows = (len(images) + cols - 1) // cols
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create figure
    fig_width = cols * 6
    fig_height = rows * 4
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create subplots
    for idx, (img_path, img) in enumerate(zip(snapshot_files, images)):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.axis('off')
        
        # Extract tick number from filename
        tick = Path(img_path).stem.split('_')[-1]
        ax.set_title(f"Tick {tick}", fontsize=10)
    
    plt.tight_layout()
    output_path = Path(snapshot_dir) / output_file
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grid saved to: {output_path}")


def create_animation(snapshot_dir, output_file='simulation.gif', fps=2):
    """
    Create an animated GIF from snapshots.
    
    Args:
        snapshot_dir: Directory containing snapshot PNG files
        output_file: Output filename for the animation
        fps: Frames per second
    """
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(str(Path(snapshot_dir) / "snapshot_tick_*.png")))
    
    if not snapshot_files:
        print(f"No snapshot files found in {snapshot_dir}")
        return
    
    print(f"Creating animation from {len(snapshot_files)} snapshots...")
    
    # Load images
    images = [Image.open(f) for f in snapshot_files]
    
    # Save as animated GIF
    output_path = Path(snapshot_dir) / output_file
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,  # Duration in milliseconds
        loop=0
    )
    
    print(f"Animation saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Create grid or animation from snapshots")
    parser.add_argument("snapshot_dir", help="Directory containing snapshots")
    parser.add_argument("--grid", action="store_true", help="Create a grid view")
    parser.add_argument("--animate", action="store_true", help="Create an animation")
    parser.add_argument("--cols", type=int, default=3, help="Columns for grid view")
    parser.add_argument("--fps", type=int, default=2, help="FPS for animation")
    
    args = parser.parse_args()
    
    if not (args.grid or args.animate):
        print("Please specify --grid or --animate (or both)")
        sys.exit(1)
    
    if args.grid:
        create_snapshot_grid(args.snapshot_dir, cols=args.cols)
    
    if args.animate:
        create_animation(args.snapshot_dir, fps=args.fps)
