"""
Batch Generator for Aquascan Simulations

This script runs multiple Aquascan simulations in parallel with different random seeds.
Each simulation produces an HDF5 file containing snapshots at configurable intervals.

Based on visualization analysis, we use 5-tick intervals to capture smooth motion while
keeping file sizes manageable (~2MB per run with 48 snapshots).

Usage:
    python -m aquascan.batch.generate --cfg configs/optimal_5tick.yml --runs 1000 --out data/raw --jobs 8

Arguments:
    --cfg: Path to the base YAML configuration file (passed to each run)
    --runs: Number of simulations to run (default: 1000)
    --out: Output directory for HDF5 files (default: data/raw)
    --jobs: Number of parallel workers (default: CPU count)
    --overwrite: If set, overwrite existing files (default: skip)
    --validate: Validate files after generation
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import hashlib

from tqdm import tqdm
from omegaconf import OmegaConf

from aquascan.run_simulation import run


def _worker(seed_cfg_pair: Tuple[int, str, Path, bool]) -> Tuple[str, Dict[str, Any]]:
    """
    Worker function to run a single simulation with the given seed.
    
    Args:
        seed_cfg_pair: Tuple of (seed, config_path, output_directory, overwrite)
        
    Returns:
        Tuple of (status, info_dict)
        - status: "ok" for success, "skip" for skipped, "error" for failure
        - info_dict: Additional information about the run
    """
    seed, cfg_yaml, out_dir, overwrite = seed_cfg_pair
    out_path = out_dir / f"{seed}.h5"
    
    # Skip if file exists and not overwriting
    if out_path.exists() and not overwrite:
        return "skip", {"seed": seed, "path": str(out_path)}
    
    try:
        # Load configuration and override seed
        cfg = OmegaConf.load(cfg_yaml)
        cfg.sim.seed = seed
        
        # Get snapshot interval (default to 5 based on visualization analysis)
        snapshot_interval = cfg.sim.get('snapshot_interval', 5)
        
        # Run simulation with x128 speed and save to file
        start_time = time.time()
        # Note: x128 speed is set inside run() before any ticks execute
        sim = run(
            ticks=cfg.sim.ticks, 
            seed=seed, 
            visual=False, 
            out_path=out_path,
            snapshot_interval=snapshot_interval
        )
        
        run_time = time.time() - start_time
        
        # Get file size
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        
        return "ok", {
            "seed": seed,
            "path": str(out_path),
            "run_time": run_time,
            "file_size_mb": file_size_mb,
            "detections": sim.stats.get("detections", 0),
            "snapshots": (cfg.sim.ticks // snapshot_interval) + 1
        }
    except Exception as e:
        return "error", {"seed": seed, "error": str(e)}


def estimate_dataset_size(runs: int, ticks: int, snapshot_interval: int) -> Dict[str, float]:
    """
    Estimate the total dataset size based on configuration.
    
    Based on visualization analysis:
    - Per snapshot: ~40KB (581 nodes + ~1200 edges)
    - Per run: snapshots * 40KB
    """
    snapshots_per_run = (ticks // snapshot_interval) + 1
    kb_per_snapshot = 40  # Based on empirical measurements
    mb_per_run = (snapshots_per_run * kb_per_snapshot) / 1024
    total_gb = (runs * mb_per_run) / 1024
    
    return {
        "snapshots_per_run": snapshots_per_run,
        "mb_per_run": mb_per_run,
        "total_gb": total_gb
    }


def validate_files(out_dir: Path, expected_count: int, sample_seed: Optional[int] = None) -> bool:
    """
    Validate the generated files.
    
    Args:
        out_dir: Directory containing HDF5 files
        expected_count: Expected number of files
        sample_seed: Seed to validate for detailed inspection
        
    Returns:
        True if validation passes, False otherwise
    """
    import h5py
    import numpy as np
    from glob import glob
    
    # Check file count
    h5_files = list(out_dir.glob("*.h5"))
    if len(h5_files) != expected_count:
        print(f"❌ Error: Expected {expected_count} files, found {len(h5_files)}")
        return False
    
    print(f"✅ Found {len(h5_files)} files as expected")
    
    # Check a sample file for schema integrity
    if h5_files and sample_seed is not None:
        sample_file = out_dir / f"{sample_seed}.h5"
        if not sample_file.exists():
            sample_file = h5_files[0]
            
        print(f"\nValidating sample file: {sample_file.name}")
        
        try:
            with h5py.File(sample_file, "r") as f:
                # Check datasets existence
                if "/nodes" not in f or "/edges" not in f or "/globals" not in f:
                    print("❌ Error: Required datasets missing")
                    return False
                
                # Check non-empty datasets
                if f["nodes"].shape[0] == 0:
                    print("❌ Error: Empty nodes dataset")
                    return False
                
                # Parse metadata
                meta = json.loads(f["globals"][()].decode())
                
                # Print file statistics
                print(f"📊 File statistics:")
                print(f"   - Nodes: {f['nodes'].shape[0]:,} entries")
                print(f"   - Edges: {f['edges'].shape[0]:,} entries")
                print(f"   - Snapshots: {meta.get('snapshot_count', 'N/A')}")
                print(f"   - Interval: every {meta.get('snapshot_interval', 1)} ticks")
                print(f"   - Simulation time: {meta['ticks']} ticks")
                print(f"   - File size: {sample_file.stat().st_size / (1024*1024):.2f} MB")
                
                # Verify snapshot interval is respected
                node_ticks = np.unique(f["nodes"]["t"])
                if len(node_ticks) > 1:
                    actual_interval = node_ticks[1] - node_ticks[0]
                    expected_interval = meta.get('snapshot_interval', 1)
                    if actual_interval != expected_interval:
                        print(f"⚠️  Warning: Expected interval {expected_interval}, got {actual_interval}")
                
        except Exception as e:
            print(f"❌ Error validating file: {e}")
            return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate batch Aquascan simulations with optimized snapshot intervals"
    )
    parser.add_argument("--cfg", default="configs/optimal_5tick.yml", help="Configuration file path")
    parser.add_argument("--runs", type=int, default=1000, help="Number of simulations to run")
    parser.add_argument("--out", default="data/raw", help="Output directory for HDF5 files")
    parser.add_argument("--jobs", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--validate", action="store_true", help="Validate files after generation")
    args = parser.parse_args()
    
    # Load configuration to get parameters
    cfg = OmegaConf.load(args.cfg)
    ticks = cfg.sim.ticks
    snapshot_interval = cfg.sim.get('snapshot_interval', 5)
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate seed range
    seeds = range(args.runs)
    
    # Estimate dataset size
    size_est = estimate_dataset_size(args.runs, ticks, snapshot_interval)
    
    # Show configuration
    print(f"🚀 [Aquascan Batch Generator - Optimized for 5-tick intervals]")
    print(f"📋 Configuration:")
    print(f"   - Config file: {args.cfg}")
    print(f"   - Output directory: {out_dir}")
    print(f"   - Simulations: {args.runs}")
    print(f"   - Parallel workers: {args.jobs}")
    print(f"   - Simulation ticks: {ticks} (@ x128 speed)")
    print(f"   - Snapshot interval: every {snapshot_interval} ticks")
    print(f"   - Snapshots per run: {size_est['snapshots_per_run']}")
    print(f"\n💾 Storage estimates:")
    print(f"   - Per run: ~{size_est['mb_per_run']:.1f} MB")
    print(f"   - Total dataset: ~{size_est['total_gb']:.2f} GB")
    
    if size_est['total_gb'] > 5.0:
        print(f"\n⚠️  Warning: Estimated size ({size_est['total_gb']:.2f} GB) exceeds 5GB target!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 0
    
    print(f"\n🏃 Starting batch generation...")
    
    # Prepare tasks
    tasks = [(s, args.cfg, out_dir, args.overwrite) for s in seeds]
    
    # Track statistics
    stats = {"ok": 0, "skip": 0, "error": 0}
    total_size_mb = 0
    total_detections = 0
    start_time = time.time()
    
    # Run tasks in parallel
    with mp.Pool(args.jobs) as pool:
        # Use tqdm for progress tracking
        with tqdm(total=len(tasks), unit="sim", desc="Simulations") as pbar:
            for status, info in pool.imap_unordered(_worker, tasks):
                # Update statistics
                stats[status] = stats.get(status, 0) + 1
                
                if status == "ok":
                    total_size_mb += info.get("file_size_mb", 0)
                    total_detections += info.get("detections", 0)
                
                # Update progress bar with detailed stats
                pbar.update(1)
                pbar.set_postfix(
                    ok=stats["ok"], 
                    skip=stats["skip"], 
                    error=stats["error"],
                    size_GB=f"{total_size_mb/1024:.2f}"
                )
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n✅ Batch generation completed in {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"\n📊 Summary:")
    print(f"   - Successful: {stats['ok']}")
    print(f"   - Skipped: {stats['skip']}")
    print(f"   - Errors: {stats['error']}")
    print(f"   - Total size: {total_size_mb/1024:.2f} GB")
    print(f"   - Avg detections/run: {total_detections/(stats['ok'] or 1):.1f}")
    print(f"   - Avg time/run: {elapsed/(stats['ok'] or 1):.2f}s")
    
    # Validate files if requested
    if args.validate:
        print("\n🔍 Validating files...")
        validation_passed = validate_files(out_dir, stats["ok"] + stats["skip"], seeds[0] if seeds else None)
        if validation_passed:
            print("✅ Validation passed!")
        else:
            print("❌ Validation failed!")
            return 1
    
    print(f"\n💡 Next steps:")
    print(f"   1. Build graph dataset: python -m aquascan.dataset.build_graphs --raw {out_dir}")
    print(f"   2. Train models on the processed graphs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
