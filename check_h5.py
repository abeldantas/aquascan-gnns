import h5py
import numpy as np
import sys
from pathlib import Path

# Set paths
raw_dir = Path("data/raw")
files = list(raw_dir.glob("*.h5"))

if not files:
    print("No HDF5 files found in data/raw")
    sys.exit(1)

# Take the first file
sample_file = files[0]
print(f"Examining file: {sample_file}")

with h5py.File(sample_file, 'r') as f:
    # Print basic information
    print("\nFile structure:")
    for key in f.keys():
        print(f"- {key}: {f[key].shape if hasattr(f[key], 'shape') else 'scalar'}")
    
    # Print data types
    print("\nData types:")
    print(f"- nodes dtype: {f['nodes'].dtype}")
    print(f"- edges dtype: {f['edges'].dtype}")
    
    # Print unique tick values
    ticks = np.unique(f['nodes']['t'])
    print(f"\nUnique ticks: {len(ticks)}")
    print(f"First few ticks: {ticks[:5]}")
    print(f"Last few ticks: {ticks[-5:] if len(ticks) >= 5 else ticks}")
    
    # Count nodes by type
    node_types = np.unique(f['nodes']['type'], return_counts=True)
    print("\nNode types:")
    for i, (t, count) in enumerate(zip(*node_types)):
        print(f"- Type {t}: {count} nodes")
    
    # Count edge relations
    edge_relations = np.unique(f['edges']['rel'], return_counts=True)
    print("\nEdge relations:")
    for i, (r, count) in enumerate(zip(*edge_relations)):
        print(f"- Relation {r}: {count} edges")
    
    # Retrieve the global metadata
    globals_data = f['globals'][()]
    if isinstance(globals_data, bytes):
        globals_data = globals_data.decode('utf-8')
    print(f"\nGlobals data: {globals_data}")
    
    # Sample a few nodes and edges
    print("\nSample nodes (first 3):")
    for i, node in enumerate(f['nodes'][:3]):
        print(f"- Node {i}: t={node['t']}, gid={node['gid']}, type={node['type']}, x={node['x']:.3f}, y={node['y']:.3f}")
    
    print("\nSample edges (first 3):")
    for i, edge in enumerate(f['edges'][:3]):
        print(f"- Edge {i}: t={edge['t']}, src={edge['src']}, dst={edge['dst']}, rel={edge['rel']}")

print("\nFile inspection complete.")
