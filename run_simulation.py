"""
Main entry point for running the Aquascan simulation.

This script initializes and runs the Aquascan simulation with visualization.
It sets up the Bokeh server and starts the simulation with the configured parameters.
"""

import argparse
import os
import sys
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

# Make sure the project root is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from visual.bokeh_app import initialize


def bokeh_app(doc):
    """Initialize the Bokeh application with the Aquascan visualization."""
    viz = initialize()
    # The viz object creates and adds all the necessary elements to the document


def main():
    """Parse arguments and start the simulation server."""
    parser = argparse.ArgumentParser(description='Run Aquascan Marine Simulation')
    parser.add_argument('--port', type=int, default=5006,
                        help='Port to run Bokeh server on (default: 5006)')
    parser.add_argument('--show', action='store_true',
                        help='Open browser automatically')
    args = parser.parse_args()
    
    print(f"Starting Aquascan simulation server on port {args.port}")
    print("Press Ctrl+C to stop")
    
    # Start Bokeh server
    server = Server({'/': bokeh_app}, port=args.port, io_loop=IOLoop.current(),
                   allow_websocket_origin=["localhost:5006"])
    
    server.start()
    
    if args.show:
        server.io_loop.add_callback(server.show, "/")
    
    server.io_loop.start()


if __name__ == "__main__":
    main()
