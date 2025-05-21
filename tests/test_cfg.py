"""
Test OmegaConf configuration for the Aquascan simulation.
"""

import pytest
from omegaconf import OmegaConf
from aquascan.run_simulation import run


def test_cfg_override():
    """Test that configuration overrides work correctly."""
    cfg = OmegaConf.create({'sim': {'ticks': 3, 'seed': 99}})
    
    # Run the simulation with the configuration
    sim = run(ticks=cfg.sim.ticks, seed=cfg.sim.seed, visual=False)
    
    # Make sure we have nodes and contacts
    assert len(sim.epsilon_nodes) > 0, "No epsilon nodes were created"
    assert len(sim.theta_contacts) > 0, "No theta contacts were created"
    
    # The simulation time should be at least the number of ticks * time step (default 1.0)
    assert sim.current_time >= cfg.sim.ticks, f"Simulation time {sim.current_time} is less than expected ticks {cfg.sim.ticks}"
    
    print(f"Simulation ran for {cfg.sim.ticks} ticks with seed {cfg.sim.seed}")
    print(f"  - Created {len(sim.epsilon_nodes)} epsilon nodes")
    print(f"  - Created {len(sim.theta_contacts)} theta contacts")
    print(f"  - Simulation time: {sim.current_time:.2f}s")


if __name__ == "__main__":
    # Run the test directly when this file is executed
    test_cfg_override()
