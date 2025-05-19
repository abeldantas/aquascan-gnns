"""
Debug entry point for running a simplified version of the Aquascan visualization.

This script initializes and runs a basic visualization of the Aquascan simulation
to help diagnose rendering issues.
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

from visual.simple_viz import initialize


def bokeh_app(doc):
    """Initialize the Bokeh application with the simple Aquascan visualization."""
    from visual.simple_viz import initialize
    
    # Get the layout from initialize() and add it to the document
    layout = initialize()
    doc.add_root(layout)
    doc.title = "Aquascan Debug Visualization"


def main():
    """Parse arguments and start the debug visualization server."""
    parser = argparse.ArgumentParser(description='Run Aquascan Debug Visualization')
    parser.add_argument('--port', type=int, default=5006,
                       help='Port to run Bokeh server on (default: 5006)')
    parser.add_argument('--show', action='store_true',
                       help='Open browser automatically')
    args = parser.parse_args()
    
    print(f"Starting Aquascan debug visualization server on port {args.port}")
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
