#!/usr/bin/env python3
"""
Redirect to the parallel version - no more suffering!
"""
import subprocess
import sys

print("🚨 WARNING: Sequential graph builder is DEPRECATED!")
print("🚀 Redirecting to parallel version...")
print()

# Replace old script path with new parallel one
new_args = sys.argv[:]
new_args[0] = "scripts/build_graphs_parallel.py"

# Run the parallel version
subprocess.run(["python3"] + new_args)
