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
    detected_contact_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'type': []
    })
    undetected_contact_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'type': []
    })
    
    # Create main plot
    plot = figure(
        width=1200, height=800,  # Increased size to fill more of the browser window
        title="",  # Remove the title
        x_range=(-3, AREA_LENGTH + 3),  # Add 3km extra view on each side
        y_range=(SHORE_DISTANCE - 3, SHORE_DISTANCE + AREA_WIDTH + 3),  # Add 3km extra view on each side
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    
    # Configure plot
    plot.grid.grid_line_color = "rgba(0, 0, 100, 0.2)"  # Light blue grid
    plot.grid.grid_line_alpha = 0.5
    plot.background_fill_color = "#e6f3ff"  # Super light blue background
    plot.border_fill_color = "#e6f3ff"  # Match the border with the background
    plot.xaxis.axis_label = "Distance along coastline (km)"
    plot.yaxis.axis_label = "Distance from shore (km) [Deployment area: 6-22 km]"
    
    # Add glyphs - use basic circle for simplicity
    # ε-nodes (blue circles) - smaller size
    plot.circle(
        x='x', y='y', source=epsilon_source,
        size=4, fill_color='#3288bd', line_color='#3288bd',
        alpha=0.7, legend_label="ε-nodes"
    )
    
    # σ-nodes (red squares) - larger size
    plot.circle(
        x='x', y='y', source=sigma_source,
        size=16, fill_color='#fc8d59', line_color='#fc8d59',
        alpha=0.9, legend_label="σ-nodes"
    )
    
    # θ-contacts detected (bright green)
    plot.circle(
        x='x', y='y', source=detected_contact_source,
        size=10, fill_color='#00cc00', line_color='#00cc00',
        alpha=0.8, legend_label="θ-contacts (detected)"
    )
    
    # θ-contacts not detected (grey)
    plot.circle(
        x='x', y='y', source=undetected_contact_source,
        size=10, fill_color='#999999', line_color='#999999',
        alpha=0.8, legend_label="θ-contacts (not detected)"
    )
    
    # Configure legend
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    
    # Add deployment area outline as a semi-transparent rectangle to highlight the area
    # This rectangle outlines the deployment area from 6-22km offshore and 0-30km along coastline
    plot.rect(
        x=AREA_LENGTH/2, y=SHORE_DISTANCE + AREA_WIDTH/2,  # Center of the rectangle 
        width=AREA_LENGTH, height=AREA_WIDTH,
        fill_color=None, line_color='#999999', line_width=2, line_dash='dashed',
        alpha=0.5
    )
    
    # Info panel
    info_div = Div(
        text=f"""
        <h3>Simulation Details</h3>
        <b>Deployment Area:</b> {AREA_LENGTH}km × {AREA_WIDTH}km<br>
        <b>Distance from Shore:</b> {SHORE_DISTANCE}km to {SHORE_DISTANCE + AREA_WIDTH}km<br>
        <b>Sensor Resolution:</b> {simulation.ocean_area.resolution}km<br>
        <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes, {len(simulation.sigma_nodes)} σ-nodes<br>
        <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
        <b>Status:</b> Ready
        """,
        width=400, height=200  # Increased height for more content
    )
    
    # Control buttons
    start_button = Button(label="Start Simulation", button_type="success")
    stop_button = Button(label="Stop Simulation", button_type="danger")
    
    # Layout
    controls = column(
        info_div,
        Div(text="<div style='height:40px;'></div>"),  # Add spacing
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
        
        # Update θ-contacts - split into detected and undetected sources
        detected_data = {
            'x': [], 'y': [], 'id': [], 'type': []
        }
        undetected_data = {
            'x': [], 'y': [], 'id': [], 'type': []
        }
        
        # For each contact, check if it's within detection range of any epsilon node
        detection_radius_km = simulation.epsilon_nodes[0].detection_radius if simulation.epsilon_nodes else 0.2
        print(f"Detection radius: {detection_radius_km}km")
        
        currently_detected = 0
        
        for contact in simulation.theta_contacts:
            is_detected = False
            detecting_nodes = []
            
            # Check if within detection radius of any epsilon node
            for node in simulation.epsilon_nodes:
                distance = np.linalg.norm(node.position - contact.position)
                if distance <= detection_radius_km:
                    is_detected = True
                    detecting_nodes.append(node.id)
            
            # Add to the appropriate data source
            if is_detected:
                detected_data['x'].append(contact.position[0])
                detected_data['y'].append(contact.position[1])
                detected_data['id'].append(contact.id)
                detected_data['type'].append(contact.type)
                currently_detected += 1
                
                # Print detection information
                print(f"Contact {contact.id} detected by nodes: {', '.join(detecting_nodes)}")
            else:
                undetected_data['x'].append(contact.position[0])
                undetected_data['y'].append(contact.position[1])
                undetected_data['id'].append(contact.id)
                undetected_data['type'].append(contact.type)
        
        # Print sample contact positions
        if len(simulation.theta_contacts) > 0:
            print(f"Sample θ-contact positions and detection status:")
            for i in range(min(5, len(simulation.theta_contacts))):
                is_detected = any(simulation.theta_contacts[i].id == id for id in detected_data['id'])
                status = "DETECTED" if is_detected else "not detected"
                print(f"  Contact {simulation.theta_contacts[i].id}: {simulation.theta_contacts[i].position} - {status}")
        
        # Update data sources
        epsilon_source.data = epsilon_data
        sigma_source.data = sigma_data
        detected_contact_source.data = detected_data
        undetected_contact_source.data = undetected_data
        
        # Update info display
        elapsed_time = simulation.current_time
        hours = int(elapsed_time / 3600)
        minutes = int((elapsed_time % 3600) / 60)
        seconds = int(elapsed_time % 60)
        
        # Count currently detected contacts
        currently_detected = sum(1 for detected in contact_data['detected'] if detected)
        
        status = "running" if simulation.is_running else "stopped"
        
        info_text = f"""
        <h3>Simulation Details</h3>
        <b>Deployment Area:</b> {AREA_LENGTH}km × {AREA_WIDTH}km<br>
        <b>Distance from Shore:</b> {SHORE_DISTANCE}km to {SHORE_DISTANCE + AREA_WIDTH}km<br>
        <b>Sensor Resolution:</b> {simulation.ocean_area.resolution}km<br>
        <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes, {len(simulation.sigma_nodes)} σ-nodes<br>
        <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
        <b>Status:</b> {status.capitalize()}<br>
        <b>Time:</b> {hours:02d}:{minutes:02d}:{seconds:02d}<br>
        <b>Total Detections:</b> {simulation.stats['detections']}<br>
        <b>Currently Detected:</b> {currently_detected} contacts<br>
        <b>Messages Delivered:</b> {simulation.stats['messages_delivered']}
        """
        info_div.text = info_text
    
    # Call update_data once to initialize
    update_data()
    
    # Button callbacks
    def start_simulation():
        simulation.start()
        update_data()
        # Increase the update frequency for more fluid animation (100ms instead of 500ms)
        doc.add_periodic_callback(simulation_tick, 100)  # 10 FPS
    
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
