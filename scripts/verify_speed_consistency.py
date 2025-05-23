#!/usr/bin/env python3
"""
Verify that batch generation uses the same x128 speed as visualization.
This script runs a single simulation both ways and compares the outputs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aquascan.simulation.simulation_loop import AquascanSimulation
from aquascan.run_simulation import run
import numpy as np

def test_speed_consistency():
    """Test that both approaches use x128 speed."""
    
    print("🔍 Verifying x128 speed consistency...\n")
    
    # Test 1: Run via snapshot generator approach
    print("Test 1: Snapshot generator approach")
    sim1 = AquascanSimulation(seed=42)
    sim1.initialize()
    sim1.start()
    sim1.set_speed(128)  # Set speed BEFORE running ticks
    
    # Run 5 ticks and record positions
    positions1 = []
    for i in range(5):
        if i > 0:
            sim1.tick()
        # Record first epsilon node position
        positions1.append(sim1.epsilon_nodes[0].position.copy())
        print(f"  Tick {i}: epsilon[0] at {positions1[-1]}, time={sim1.current_time:.1f}s")
    
    # Test 2: Run via batch generator approach (using run function)
    print("\nTest 2: Batch generator approach (via run function)")
    from pathlib import Path
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
        sim2 = run(ticks=4, seed=42, visual=False, out_path=Path(tmp.name), snapshot_interval=1)
        
        # Check positions from the completed simulation
        # The simulation has already run, so we need to check the stored data
        import h5py
        with h5py.File(tmp.name, 'r') as f:
            nodes = f['nodes'][:]
            # Get first epsilon node (gid=0) positions at each tick
            positions2 = []
            for t in range(5):
                tick_nodes = nodes[nodes['t'] == t]
                epsilon0 = tick_nodes[tick_nodes['gid'] == 0][0]
                pos = np.array([epsilon0['x'], epsilon0['y']])
                positions2.append(pos)
                print(f"  Tick {t}: epsilon[0] at {pos}, time={t*128:.1f}s")
    
    # Test 3: Verify time progression
    print(f"\nTest 3: Time progression check")
    print(f"  Snapshot approach final time: {sim1.current_time:.1f}s")
    print(f"  Batch approach expected time: {4 * 128:.1f}s (4 ticks × 128s)")
    print(f"  Speed factor in sim1: {sim1.speed_factor}x")
    
    # Compare positions
    print("\n📊 Position comparison:")
    all_match = True
    for i in range(5):
        diff = np.linalg.norm(positions1[i] - positions2[i])
        match = "✅" if diff < 1e-6 else "❌"
        print(f"  Tick {i}: difference = {diff:.6f} km {match}")
        if diff >= 1e-6:
            all_match = False
    
    if all_match:
        print("\n✅ SUCCESS: Both approaches use identical x128 speed!")
        print("   The batch generator produces the exact same simulation as the visualizations.")
    else:
        print("\n❌ MISMATCH: The approaches differ!")
        print("   This would indicate a problem with the speed setting.")
    
    # Additional verification: Check simulation stats
    print(f"\n📈 Simulation stats:")
    print(f"  Snapshot approach detections: {sim1.stats['detections']}")
    print(f"  Batch approach detections: {sim2.stats['detections']}")

if __name__ == "__main__":
    test_speed_consistency()
