#!/usr/bin/env python3
"""
Parallelized graph builder with proper progress tracking.
Run this instead of the slow sequential version!
"""

import os
import sys
import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aquascan.dataset.build_graphs import h5_to_graphs, save_split, validate_graphs
import torch
import numpy as np
import random


def process_single_file(args_tuple):
    """Process a single file (for multiprocessing)."""
    file_path, context_len, horizon_len = args_tuple
    try:
        graphs = h5_to_graphs(str(file_path), context_len, horizon_len)
        return (True, graphs, len(graphs))
    except Exception as e:
        return (False, str(e), 0)


def build_graphs_parallel(raw_dir, out_dir, context_len, horizon_len, 
                         split_ratios, adv_fraction=0.05, jobs=None, seed=42):
    """Build graphs in parallel with progress tracking."""
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find HDF5 files
    raw_path = Path(raw_dir)
    files = list(raw_path.glob('*.h5'))
    
    if not files:
        print(f"❌ No HDF5 files found in {raw_dir}")
        return False
    
    print(f"\n🔥 Building graphs for horizon={horizon_len} ticks")
    print(f"📂 Found {len(files)} HDF5 files in {raw_dir}")
    print(f"⚡ Using {jobs or cpu_count()} parallel workers")
    
    # Prepare arguments for parallel processing
    file_args = [(f, context_len, horizon_len) for f in files]
    
    # Process files in parallel with progress bar
    all_graphs = []
    failed_files = []
    total_windows = 0
    
    with Pool(processes=jobs) as pool:
        # Use tqdm for progress tracking
        with tqdm(total=len(files), desc="Processing files", unit="file") as pbar:
            for success, result, num_graphs in pool.imap_unordered(process_single_file, file_args):
                if success:
                    all_graphs.extend(result)
                    total_windows += num_graphs
                    pbar.set_postfix({"graphs": total_windows})
                else:
                    failed_files.append(result)
                pbar.update(1)
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed to process")
    
    print(f"\n✅ Created {len(all_graphs)} graphs total")
    
    if len(all_graphs) > 0:
        print("🔍 Validating graphs...")
        if validate_graphs(all_graphs):
            print("✅ All graphs passed validation")
            
            # Save the splits
            print(f"💾 Saving to {out_dir}...")
            save_split(all_graphs, split_ratios, out_dir, adv_fraction)
            print(f"✅ Saved graph splits to {out_dir}")
            return True
        else:
            print("❌ Graph validation failed")
            return False
    else:
        print("❌ No graphs were created")
        return False


def main():
    parser = argparse.ArgumentParser(description='Parallel Graph Builder')
    parser.add_argument('--raw', type=str, required=True, help='Input directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--context', type=int, default=60, help='Context length')
    parser.add_argument('--horizon', type=int, required=True, help='Horizon length')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument('--adv_fraction', type=float, default=0.05)
    parser.add_argument('--jobs', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    success = build_graphs_parallel(
        args.raw, args.out, args.context, args.horizon,
        args.split, args.adv_fraction, args.jobs, args.seed
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
