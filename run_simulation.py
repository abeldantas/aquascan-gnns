"""
Main entry point for running the Aquascan simulation.

This script initializes and runs the Aquascan simulation with visualization.
It sets up the Bokeh server and starts the simulation with the configured parameters.
"""

import argparse
import os
import sys
import time
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
        VIZ_UPDATE_INTERVAL, PLOT_WIDTH, PLOT_HEIGHT, SHORE_DISTANCE,
        SIMULATION_SPEED, SIMULATION_START_HOUR
    )
    import numpy as np
    from bokeh.plotting import figure
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource, Button, Div, Slider, Select
    
    # Create simulation instance
    simulation = AquascanSimulation()
    simulation.initialize()
    
    # Update data sources for visualization
    epsilon_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'color': []
    })
    sigma_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'color': []
    })
    detected_contact_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'type': [], 'label': []
    })
    undetected_contact_source = ColumnDataSource({
        'x': [], 'y': [], 'id': [], 'type': [], 'label': []
    })
    
    # Create main plot
    plot = figure(
        width=1200, height=800,  # Increased size to fill more of the browser window
        title="",  # Remove the title
        x_range=(-3, AREA_LENGTH + 3),  # Add 3km extra view on each side
        y_range=(SHORE_DISTANCE - 3, SHORE_DISTANCE + AREA_WIDTH + 3),  # Add 3km extra view on each side
        tools="pan,wheel_zoom,box_zoom,reset,save",
        margin=(30, 10, 10, 30)  # Top, right, bottom, left margins in pixels
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
    plot.scatter(
        x='x', y='y', source=epsilon_source,
        size=4, fill_color='#3288bd', line_color='#3288bd',
        alpha=0.7, marker="circle"
    )
    
    # σ-nodes (red squares) - larger size
    plot.scatter(
        x='x', y='y', source=sigma_source,
        size=16, fill_color='#fc8d59', line_color='#fc8d59',
        alpha=0.9, marker="circle"
    )
    
    # θ-contacts detected (bright green)
    plot.scatter(
        x='x', y='y', source=detected_contact_source,
        size=10, fill_color='#00cc00', line_color='#00cc00',
        alpha=0.8, marker="circle"
    )
    
    # θ-contacts not detected (grey) - no legend
    plot.scatter(
        x='x', y='y', source=undetected_contact_source,
        size=10, fill_color='#999999', line_color='#999999',
        alpha=0.8, marker="circle"
    )
    
    # Add contact ID labels above detected contacts
    labels_detected = plot.text(
        x='x', y='y', text='label',
        x_offset=0, y_offset=-17,  # Position below the point instead of above
        text_font_size='8pt', text_color='#00cc00',
        text_align='center', text_baseline='top',  # Changed to top baseline
        source=detected_contact_source
    )
    
    # Add contact ID labels above undetected contacts
    labels_undetected = plot.text(
        x='x', y='y', text='label',
        x_offset=0, y_offset=-17,  # Position below the point instead of above
        text_font_size='8pt', text_color='#999999',
        text_align='center', text_baseline='top',  # Changed to top baseline
        source=undetected_contact_source
    )
    
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
        <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes (blue), {len(simulation.sigma_nodes)} σ-nodes (red)<br>
        <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
        <b>Status:</b> Ready
        
        <h4>Entity Types:</h4>
        <ul style="padding-left: 20px; margin-top: 5px;">
          <li>Detected θ-contacts: <span style="color: #00cc00;">●</span> (bright green)</li>
          <li>Undetected θ-contacts: <span style="color: #999999;">●</span> (grey)</li>
        </ul>
        
        <h4>Marine Species:</h4>
        <div id="contacts-list" style="max-height: 150px; overflow-y: auto; margin-top: 5px;">
        </div>
        """,
        width=400
    )
    
    # Control buttons
    start_button = Button(label="Start Simulation", button_type="success", width=150)
    stop_button = Button(label="Stop Simulation", button_type="danger", width=150)
    
    # Speed control buttons
    speed_realtime = Button(label="Realtime", button_type="default", width=75)
    speed_x4 = Button(label="x4", button_type="default", width=50)
    speed_x8 = Button(label="x8", button_type="default", width=50)
    speed_x16 = Button(label="x16", button_type="default", width=50)
    speed_x32 = Button(label="x32", button_type="default", width=50)
    speed_x64 = Button(label="x64", button_type="default", width=50)
    speed_x128 = Button(label="x128", button_type="default", width=50)
    speed_x256 = Button(label="x256", button_type="default", width=50)

    # Control headers
    control_header = Div(text="<h4>Simulation Controls:</h4>", margin=(15, 0, 5, 0))
    speed_header = Div(text="<h4>Simulation Speed:</h4>", margin=(15, 0, 5, 0))
    
    # Group the controls in sections with padding
    simulation_buttons = row(start_button, stop_button, margin=(0, 0, 10, 0))
    speed_buttons = row(speed_realtime, speed_x4, speed_x8, speed_x16, 
                    speed_x32, speed_x64, speed_x128, speed_x256, 
                    margin=(0, 0, 15, 0))
    
    # Create the main control column with proper spacing and fixed height
    controls = column(
        info_div,
        Div(height=20),  # Add spacing
        control_header,
        simulation_buttons,
        Div(height=10),  # Add spacing
        speed_header,
        speed_buttons,
        sizing_mode="stretch_width",
        height=600  # Fixed height to prevent overlap
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
        
        # Update σ-nodes
        sigma_data = {
            'x': [], 'y': [], 'id': [], 'color': []
        }
        for node in simulation.sigma_nodes:
            sigma_data['x'].append(node.position[0])
            sigma_data['y'].append(node.position[1])
            sigma_data['id'].append(node.id)
            sigma_data['color'].append('red')
        
        # Update θ-contacts - split into detected and undetected sources
        detected_data = {
            'x': [], 'y': [], 'id': [], 'type': [], 'label': []
        }
        undetected_data = {
            'x': [], 'y': [], 'id': [], 'type': [], 'label': []
        }
        
        # For each contact, check if it's within detection range of any epsilon node
        detection_radius_km = simulation.epsilon_nodes[0].detection_radius if simulation.epsilon_nodes else 0.2
        
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
                # Extract ID number from the full ID (e.g., "θ-001" -> "001")
                id_number = contact.id.split('-')[1]
                detected_data['label'].append(f"θ-{id_number}")
                currently_detected += 1
            else:
                undetected_data['x'].append(contact.position[0])
                undetected_data['y'].append(contact.position[1])
                undetected_data['id'].append(contact.id)
                undetected_data['type'].append(contact.type)
                # Extract ID number from the full ID
                id_number = contact.id.split('-')[1]
                undetected_data['label'].append(f"θ-{id_number}")
        
        # Update data sources
        epsilon_source.data = epsilon_data
        sigma_source.data = sigma_data
        detected_contact_source.data = detected_data
        undetected_contact_source.data = undetected_data
        
        # Calculate real elapsed time
        real_elapsed_time = time.time() - simulation.start_real_time if hasattr(simulation, 'start_real_time') else 0
        real_hours = int(real_elapsed_time / 3600)
        real_minutes = int((real_elapsed_time % 3600) / 60)
        real_seconds = int(real_elapsed_time % 60)
        
        # Calculate simulation time (starting at 9am on day 1)
        sim_elapsed_time = simulation.current_time
        start_hour = SIMULATION_START_HOUR
        
        # Convert to hours, minutes, seconds
        total_seconds = start_hour * 3600 + sim_elapsed_time
        sim_days = int(total_seconds / (24 * 3600))
        sim_hours = int((total_seconds % (24 * 3600)) / 3600)
        sim_minutes = int((total_seconds % 3600) / 60)
        sim_seconds = int(total_seconds % 60)
        
        # Count currently detected contacts
        currently_detected = len(detected_data['id'])
        
        status = "running" if simulation.is_running else "stopped"
        
        # Build contacts list HTML
        contacts_list_html = "<ul style='padding-left: 20px; margin-top: 0;'>"
        
        # Sort contacts by ID to ensure consistent ordering
        sorted_contacts = sorted(simulation.theta_contacts, key=lambda x: x.id)
        
        for contact in sorted_contacts:
            # Check if detected
            is_detected = any(contact.id == id for id in detected_data['id'])
            status_color = "#00cc00" if is_detected else "#999999"
            status_text = "Detected" if is_detected else "Not detected"
            
            # Extract contact number from ID
            contact_num = contact.id.split('-')[1]
            
            contacts_list_html += f"""
            <li>
              <span style="color: {status_color};">●</span> <b>θ-{contact_num}</b>: {contact.species_name} ({status_text})
            </li>
            """
        
        contacts_list_html += "</ul>"
        
        info_text = f"""
        <h3>Simulation Details</h3>
        <b>Deployment Area:</b> {AREA_LENGTH}km × {AREA_WIDTH}km<br>
        <b>Distance from Shore:</b> {SHORE_DISTANCE}km to {SHORE_DISTANCE + AREA_WIDTH}km<br>
        <b>Sensor Resolution:</b> {simulation.ocean_area.resolution}km<br>
        <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes (blue), {len(simulation.sigma_nodes)} σ-nodes (red)<br>
        <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
        <b>Status:</b> {status.capitalize()}<br>
        <b>Real Time:</b> {real_hours:02d}:{real_minutes:02d}:{real_seconds:02d}<br>
        <b>Simulation Time:</b> Day {sim_days+1}, {sim_hours:02d}:{sim_minutes:02d}:{sim_seconds:02d}<br>
        <b>Time Scale:</b> {simulation.speed_factor}x real-time<br>
        <b>Total Detections:</b> {simulation.stats['detections']}<br>
        <b>Currently Detected:</b> {currently_detected} contacts<br>
        <b>Messages Delivered:</b> {simulation.stats['messages_delivered']}
        
        <h4>Entity Types:</h4>
        <ul style="padding-left: 20px; margin-top: 5px;">
          <li>Detected θ-contacts: <span style="color: #00cc00;">●</span> (bright green)</li>
          <li>Undetected θ-contacts: <span style="color: #999999;">●</span> (grey)</li>
        </ul>
        
        <h4>Marine Species:</h4>
        <div id="contacts-list" style="max-height: 150px; overflow-y: auto; margin-top: 5px;">
        {contacts_list_html}
        </div>
        """
        info_div.text = info_text
    
    # Call update_data once to initialize
    update_data()
    
    # Update the currently selected speed button
    def update_speed_buttons(current_speed):
        """Update the visual state of speed buttons and the simulation speed."""
        # Update button styles
        speed_realtime.button_type = "success" if current_speed == 1 else "default"
        speed_x4.button_type = "success" if current_speed == 4 else "default"
        speed_x8.button_type = "success" if current_speed == 8 else "default"
        speed_x16.button_type = "success" if current_speed == 16 else "default"
        speed_x32.button_type = "success" if current_speed == 32 else "default"
        speed_x64.button_type = "success" if current_speed == 64 else "default"
        speed_x128.button_type = "success" if current_speed == 128 else "default"
        speed_x256.button_type = "success" if current_speed == 256 else "default"

    def set_simulation_speed(speed):
        """Set the simulation speed and update the UI."""
        simulation.set_speed(speed)
        update_speed_buttons(speed)
        update_data()
        
    update_speed_buttons(simulation.speed_factor)
    
    # Button callbacks
    def start_simulation():
        simulation.start()
        simulation.start_real_time = time.time()
        update_data()
        doc.add_periodic_callback(simulation_tick, 30)  # ~33 FPS
    
    def stop_simulation():
        simulation.stop()
        update_data()
        for callback in list(doc.session_callbacks):
            doc.remove_periodic_callback(callback)
    
    def set_speed_realtime():
        set_simulation_speed(1)
    
    def set_speed_x4():
        set_simulation_speed(4)
    
    def set_speed_x8():
        set_simulation_speed(8)
    
    def set_speed_x16():
        set_simulation_speed(16)
    
    def set_speed_x32():
        set_simulation_speed(32)
    
    def set_speed_x64():
        set_simulation_speed(64)

    def set_speed_x128():
        set_simulation_speed(128)

    def set_speed_x256():
        set_simulation_speed(256)
    
    def simulation_tick():
        if simulation.is_running:
            simulation.tick()
            update_data()
        else:
            for callback in list(doc.session_callbacks):
                doc.remove_periodic_callback(callback)
    
    # Connect callbacks
    start_button.on_click(start_simulation)
    stop_button.on_click(stop_simulation)
    speed_realtime.on_click(set_speed_realtime)
    speed_x4.on_click(set_speed_x4)
    speed_x8.on_click(set_speed_x8)
    speed_x16.on_click(set_speed_x16)
    speed_x32.on_click(set_speed_x32)
    speed_x64.on_click(set_speed_x64)
    speed_x128.on_click(set_speed_x128)
    speed_x256.on_click(set_speed_x256)
    
    # Set initial speed button style using the instance property
    update_speed_buttons(simulation.speed_factor)
    
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
