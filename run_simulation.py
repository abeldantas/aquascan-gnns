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
from bokeh.layouts import row
from bokeh.plotting import curdoc

# Make sure the project root is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def bokeh_app(doc):
    """Initialize the Bokeh application with the Aquascan visualization."""
    # Import here to avoid circular imports
    from simulation.simulation_loop import AquascanSimulation
    from config.simulation_config import (
        AREA_LENGTH, AREA_WIDTH, ENTITY_COLORS, TIME_STEP,
        VIZ_UPDATE_INTERVAL, PLOT_WIDTH, PLOT_HEIGHT, SHORE_DISTANCE
    )
    import numpy as np
    from bokeh.plotting import figure
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource, Button, Div, Slider, Select
    
    # Create simulation instance
    simulation = AquascanSimulation()
    simulation.initialize()
    
    # Data sources for Bokeh
    epsilon_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'color': []
    })
    sigma_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'color': []
    })
    contact_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'type': [], 'color': []
    })
    
    # Create main plot
    plot = figure(
        width=1200, height=800,  # Increased size to fill more of the browser window
        title="Aquascan Marine Simulation",
        x_range=(-1, AREA_LENGTH + 1),
        y_range=(SHORE_DISTANCE - 1, SHORE_DISTANCE + AREA_WIDTH + 1),  # Adjust y_range to reflect the actual area
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    
    # Configure plot
    plot.grid.grid_line_color = "rgba(255, 255, 255, 0.3)"  # Lighter grid lines
    plot.grid.grid_line_alpha = 0.7
    plot.background_fill_color = "#4D629B"  # Blue background as requested
    plot.border_fill_color = "#4D629B"
    plot.xaxis.axis_label = "Distance along coastline (km)"
    plot.yaxis.axis_label = "Distance from shore (km) [Deployment area: 6-22 km]"
    plot.xaxis.axis_label_text_color = "white"
    plot.yaxis.axis_label_text_color = "white"
    plot.xaxis.major_label_text_color = "white"
    plot.yaxis.major_label_text_color = "white"
    plot.title.text_color = "white"
    
    # Add glyphs - use basic circle for simplicity
    # ε-nodes (blue circles) - smaller size
    plot.circle(
        x='x', y='y', source=epsilon_source,
        size=5, fill_color='#3288bd', line_color='#3288bd',
        alpha=0.7, legend_label="ε-nodes"
    )
    
    # σ-nodes (white circles) - larger size
    plot.circle(
        x='x', y='y', source=sigma_source,
        size=20, fill_color='white', line_color='white',
        alpha=0.9, legend_label="σ-nodes"
    )
    
    # θ-contacts (green triangles via specific markers)
    plot.circle(
        x='x', y='y', source=contact_source,
        size=10, fill_color='#d53e4f', line_color='#d53e4f',
        alpha=0.8, legend_label="θ-contacts"
    )
    
    # Configure legend
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    
    # Info panel with white text for dark background
    info_div = Div(
        text=f"""
        <div style="color: white;">
        <h3>Aquascan Simulation</h3>
        <b>Deployment Area:</b> {AREA_LENGTH}km × {AREA_WIDTH}km<br>
        <b>Distance from Shore:</b> {SHORE_DISTANCE}km to {SHORE_DISTANCE + AREA_WIDTH}km<br>
        <b>Sensor Resolution:</b> {simulation.ocean_area.resolution}km<br>
        <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes, {len(simulation.sigma_nodes)} σ-nodes<br>
        <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
        <b>Status:</b> Ready
        </div>
        """,
        width=400, height=200  # Increased height for more content
    )
    
    # Control buttons
    start_button = Button(label="Start Simulation", button_type="success", width=180)
    stop_button = Button(label="Stop Simulation", button_type="danger", width=180)
    
    # Layout
    controls = column(
        info_div,
        row(start_button, stop_button)
    )
    
    # Update data function
    def update_data():
        # Update ε-nodes
        epsilon_data = {
            'x': [], 'y': [], 'id': [], 'color': []
        }
        for node in simulation.epsilon_nodes:
            epsilon_data['x'].append(node.position[0])
            epsilon_data['y'].append(node.position[1])
            epsilon_data['id'].append(node.id)
            epsilon_data['color'].append('blue')
        
        # Print sample positions for debugging
        if len(simulation.epsilon_nodes) > 0:
            print(f"Sample ε-node positions:")
            for i in range(min(5, len(simulation.epsilon_nodes))):
                print(f"  Node {simulation.epsilon_nodes[i].id}: {simulation.epsilon_nodes[i].position}")
        
        # Update σ-nodes
        sigma_data = {
            'x': [], 'y': [], 'id': [], 'color': []
        }
        for node in simulation.sigma_nodes:
            sigma_data['x'].append(node.position[0])
            sigma_data['y'].append(node.position[1])
            sigma_data['id'].append(node.id)
            sigma_data['color'].append('red')
        
        # Print σ-node positions
        print(f"σ-node positions:")
        for node in simulation.sigma_nodes:
            print(f"  Node {node.id}: {node.position}")
        
        # Update θ-contacts
        contact_data = {
            'x': [], 'y': [], 'id': [], 'type': [], 'color': []
        }
        for contact in simulation.theta_contacts:
            contact_data['x'].append(contact.position[0])
            contact_data['y'].append(contact.position[1])
            contact_data['id'].append(contact.id)
            contact_data['type'].append(contact.type)
            contact_data['color'].append('green')
        
        # Print sample contact positions
        if len(simulation.theta_contacts) > 0:
            print(f"Sample θ-contact positions:")
            for i in range(min(5, len(simulation.theta_contacts))):
                print(f"  Contact {simulation.theta_contacts[i].id}: {simulation.theta_contacts[i].position}")
        
        # Update data sources
        epsilon_source.data = epsilon_data
        sigma_source.data = sigma_data
        contact_source.data = contact_data
        
        # Update info display
        elapsed_time = simulation.current_time
        hours = int(elapsed_time / 3600)
        minutes = int((elapsed_time % 3600) / 60)
        seconds = int(elapsed_time % 60)
        
        status = "running" if simulation.is_running else "stopped"
        
        info_text = f"""
        <div style="color: white;">
        <h3>Aquascan Simulation</h3>
        <b>Deployment Area:</b> {AREA_LENGTH}km × {AREA_WIDTH}km<br>
        <b>Distance from Shore:</b> {SHORE_DISTANCE}km to {SHORE_DISTANCE + AREA_WIDTH}km<br>
        <b>Sensor Resolution:</b> {simulation.ocean_area.resolution}km<br>
        <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes, {len(simulation.sigma_nodes)} σ-nodes<br>
        <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
        <b>Status:</b> {status.capitalize()}<br>
        <b>Time:</b> {hours:02d}:{minutes:02d}:{seconds:02d}<br>
        <b>Detections:</b> {simulation.stats['detections']}<br>
        <b>Messages Delivered:</b> {simulation.stats['messages_delivered']}
        </div>
        """
        info_div.text = info_text
    
    # Call update_data once to initialize
    update_data()
    
    # Button callbacks
    def start_simulation():
        simulation.start()
        update_data()
        doc.add_periodic_callback(simulation_tick, 500)  # 500ms interval
    
    def stop_simulation():
        simulation.stop()
        update_data()
        # Remove periodic callback if exists
        for callback in list(doc.session_callbacks):
            doc.remove_periodic_callback(callback)
    
    def simulation_tick():
        if simulation.is_running:
            simulation.tick()
            update_data()
        else:
            # Remove callback if simulation is not running
            for callback in list(doc.session_callbacks):
                doc.remove_periodic_callback(callback)
    
    # Connect callbacks
    start_button.on_click(start_simulation)
    stop_button.on_click(stop_simulation)
    
    # Add to document (this is key - we add to the document passed to this function)
    layout = row(plot, controls)
    doc.add_root(layout)
    doc.title = "Aquascan Marine Simulation"


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
                   allow_websocket_origin=[f"localhost:{args.port}"])
    
    server.start()
    
    if args.show:
        server.io_loop.add_callback(server.show, "/")
    
    server.io_loop.start()


if __name__ == "__main__":
    main()
