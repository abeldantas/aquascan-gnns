#!/usr/bin/env python3
"""
Example script showing different visualization scenarios for Aquascan simulations.
"""

import subprocess
import sys
from pathlib import Path

def run_example(name, description, command):
    """Run an example visualization command."""
    print(f"\n{'='*60}")
    print(f"Example: {name}")
    print(f"Description: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    response = input("Run this example? (y/n): ")
    if response.lower() == 'y':
        subprocess.run(command, shell=True)
        print("✓ Completed")
    else:
        print("⏭ Skipped")

def main():
    """Run various visualization examples."""
    print("Aquascan Visualization Examples")
    print("This script demonstrates different ways to visualize the simulation.")
    
    examples = [
        (
            "Quick Test",
            "Generate snapshots for a 2-minute simulation at x128 speed",
            "./test_snapshots.sh"
        ),
        (
            "High-Frequency Snapshots", 
            "Capture every 10 seconds to see detailed movement",
            "python scripts/generate_snapshots.py --seed 42 --ticks 300 --interval 10"
        ),
        (
            "Long Simulation",
            "10-minute simulation with snapshots every minute",
            "python scripts/generate_snapshots.py --seed 42 --ticks 600 --interval 60"
        ),
        (
            "Different Seeds Comparison",
            "Run multiple simulations with different random seeds",
            """
for seed in 42 43 44; do
    python scripts/generate_snapshots.py --seed $seed --ticks 120 --interval 30 --output visualizations/seed_$seed
done
"""
        ),
        (
            "Create Grid View",
            "Combine snapshots into a single grid image",
            "python scripts/visualize_snapshots.py visualizations/run_42_* --grid --cols 4"
        ),
        (
            "Create Animation",
            "Generate animated GIF from snapshots",
            "python scripts/visualize_snapshots.py visualizations/run_42_* --animate --fps 4"
        ),
        (
            "Full Analysis Pipeline",
            "Generate data, create snapshots, and build visualization summary",
            """
# 1. Run simulation and generate snapshots
python scripts/generate_snapshots.py --seed 123 --ticks 300 --interval 20

# 2. Find the output directory
OUTPUT_DIR=$(ls -td visualizations/run_123_* | head -1)

# 3. Create both grid and animation
python scripts/visualize_snapshots.py $OUTPUT_DIR --grid --cols 3
python scripts/visualize_snapshots.py $OUTPUT_DIR --animate --fps 3

echo "Results in: $OUTPUT_DIR"
"""
        )
    ]
    
    print(f"\nFound {len(examples)} examples to demonstrate.\n")
    
    for name, desc, cmd in examples:
        run_example(name, desc, cmd)
    
    print("\n✨ Examples completed!")
    print("\nTips:")
    print("- Snapshots are saved in timestamped directories under 'visualizations/'")
    print("- Use higher --interval values for longer simulations to reduce file count")
    print("- Grid views are great for comparing states at different times")
    print("- Animations help visualize entity movement patterns")
    print("- Different seeds show variability in marine entity behavior")

if __name__ == "__main__":
    main()
