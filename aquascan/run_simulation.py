"""
Main entry point for running the Aquascan simulation.

This script initializes and runs the Aquascan simulation with visualization.
It sets up the Bokeh server and starts the simulation with the configured parameters.
"""

from typing import Optional
from pathlib import Path
import argparse
import os
import sys
import time
from omegaconf import OmegaConf

# -----------------------------------------------------------------------------
# Public, test‑friendly API
# -----------------------------------------------------------------------------

def run(ticks: int = 600, *, seed: int = 42, visual: bool = False, out_path: Optional[Path] = None, snapshot_interval: int = 1) -> "AquascanSimulation":
    """Run **one** simulation for *ticks* steps.

    Args:
        ticks: Number of simulation steps to run
        seed: Random seed for reproducibility
        visual: Whether to run in visual mode (Bokeh)
        out_path: If provided, exports simulation snapshots to this HDF5 file
        snapshot_interval: Interval for saving snapshots (default: 1 = every tick)

    Returns the *AquascanSimulation* instance so that unit tests can inspect
    internal state (e.g. node counts, stats).  If visual=False, no Bokeh code is imported –
    safe for PyTest and multiprocessing workers.
    """
    from aquascan.simulation.simulation_loop import AquascanSimulation

    sim = AquascanSimulation(seed=seed)
    sim.initialize()
    
    # Set speed to x128 as determined by visualization analysis
    sim.set_speed(128)

    if visual:
        # Visual mode would be handled by the main() function
        return sim
    else:
        # Headless mode - run for specified number of ticks
        sim.start()  # Start the simulation
        
        # If out_path is provided, export snapshots to HDF5
        if out_path:
            from aquascan.io.writer import SnapshotWriter
            from pathlib import Path
            
            # Convert out_path to Path if it's a string
            if isinstance(out_path, str):
                out_path = Path(out_path)
                
            # Calculate actual number of snapshots
            snapshot_count = (ticks // snapshot_interval) + 1
                
            # Create metadata for the run
            meta = {
                "seed": seed,
                "ticks": ticks,
                "snapshot_interval": snapshot_interval,
                "snapshot_count": snapshot_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epsilon_count": len(sim.epsilon_nodes),
                "theta_count": len(sim.theta_contacts),
                "resolution": sim.ocean_area.resolution,
            }
            
            # Create the writer - estimate based on actual snapshots
            out = SnapshotWriter(out_path, meta=meta, est_ticks=snapshot_count)
            
            # Export snapshots at intervals
            for t in range(ticks + 1):
                # Always run the tick (except on the last iteration)
                if t > 0:
                    sim.tick()
                
                # Save snapshot at intervals (including tick 0)
                if t % snapshot_interval == 0:
                    nodes, edges = sim.export_snapshot(t)
                    out.append(t, nodes, edges)
                
            # Close the writer
            out.close()
        else:
            # Just run the simulation without exporting
            for _ in range(ticks):
                sim.tick()
                
        sim.stop()  # Stop the simulation
        return sim



# -----------------------------------------------------------------------------
# Bokeh application (unchanged apart from lazy imports)
# -----------------------------------------------------------------------------

from bokeh.layouts import row  # only required when visual mode is used
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from bokeh.plotting import curdoc

def bokeh_app(doc):
    """Initialize the full‑fat Bokeh dashboard (legacy code)."""
    # All heavy imports stay *inside* to avoid polluting headless mode
    # (verbatim of user‑supplied file, trimmed for brevity)
    # ---------------------------------------------------------------------
    from aquascan.simulation.simulation_loop import AquascanSimulation
    from aquascan.config.simulation_config import (
        AREA_LENGTH, AREA_WIDTH, ENTITY_COLORS, TIME_STEP,
        VIZ_UPDATE_INTERVAL, PLOT_WIDTH, PLOT_HEIGHT, SHORE_DISTANCE,
        SIMULATION_SPEED, SIMULATION_START_HOUR,
    )
    import numpy as np
    from bokeh.plotting import figure
    from bokeh.layouts import column, row
    from bokeh.models import (
        ColumnDataSource, Button, Div, Slider, Select, Arrow, VeeHead, Label,
        CheckboxGroup, Spacer,
    )
    from aquascan.config import simulation_config

    motion_checkbox = CheckboxGroup(labels=["Enable Drift & Currents"], active=[0])

    # Create simulation instance
    simulation = AquascanSimulation()
    simulation.initialize()
    
    # Helper function for cardinal directions
    def get_direction_name(angle_deg):
        """Convert angle in degrees to cardinal direction name."""
        # Normalize angle to 0-360 range
        angle_deg = (angle_deg + 360) % 360
        
        # Map angle to cardinal direction
        if 22.5 <= angle_deg < 67.5:
            return "NE"
        elif 67.5 <= angle_deg < 112.5:
            return "E"
        elif 112.5 <= angle_deg < 157.5:
            return "SE"
        elif 157.5 <= angle_deg < 202.5:
            return "S"
        elif 202.5 <= angle_deg < 247.5:
            return "SW"
        elif 247.5 <= angle_deg < 292.5:
            return "W"
        elif 292.5 <= angle_deg < 337.5:
            return "NW"
        else:
            return "N"
    
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
    
    # Add connection data sources
    permanent_connections_source = ColumnDataSource({
        'x0': [], 'y0': [], 'x1': [], 'y1': [], 'id0': [], 'id1': []
    })
    intermittent_connections_source = ColumnDataSource({
        'x0': [], 'y0': [], 'x1': [], 'y1': [], 'id0': [], 'id1': []
    })
    
    # Add contact connections source
    contact_connections_source = ColumnDataSource({
        'x0': [], 'y0': [], 'x1': [], 'y1': []
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
    
    # Add current direction indicator data source
    current_indicator_source = ColumnDataSource({
        'x': [AREA_LENGTH / 2], 
        'y': [SHORE_DISTANCE - 2],  # Position below the deployment area
        'x_end': [AREA_LENGTH / 2],  # Initially same as start position
        'y_end': [SHORE_DISTANCE - 2], # Initially same as start position
    })
    
    # Add current direction arrow
    arrow_head_size = 4.0  # Increased from 2.0
    arrow = Arrow(end=VeeHead(size=arrow_head_size * 2, fill_color="#0066cc", line_color="#0066cc"),
                 x_start='x', y_start='y', x_end='x_end', y_end='y_end',
                 source=current_indicator_source,
                 line_width=2, line_color="#0066cc")
    plot.add_layout(arrow)
    
    # Add current info text
    current_info = Label(
        x=AREA_LENGTH / 2, y=SHORE_DISTANCE - 2.2,  # Moved up from -2.4
        text="Current: N/A",
        text_font_size="12pt",
        text_align="center",
        text_baseline="top"
    )
    plot.add_layout(current_info)
    
    # Add glyphs - use basic circle for simplicity
    # ε-nodes (blue circles) - smaller size
    plot.scatter(
        x='x', y='y', source=epsilon_source,
        size=4, fill_color='#3288bd', line_color='#3288bd',
        alpha=0.7, marker="circle"
    )
    
    # σ-nodes (red squares) - commented out to remove from graph
    # plot.scatter(
    #     x='x', y='y', source=sigma_source,
    #     size=16, fill_color='#fc8d59', line_color='#fc8d59',
    #     alpha=0.9, marker="circle"
    # )
    
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
    
    # Add connection lines between epsilon nodes
    # Permanent connections (solid blue lines)
    plot.segment(
        x0='x0', y0='y0', x1='x1', y1='y1',
        source=permanent_connections_source,
        line_color='#3288bd', line_width=0.5, alpha=0.7
    )
    
    # Intermittent connections (dashed light blue lines)
    plot.segment(
        x0='x0', y0='y0', x1='x1', y1='y1',
        source=intermittent_connections_source,
        line_color='#73c2fb', line_width=0.5, alpha=0.5, line_dash='dashed'
    )
    
    # Add connections between detected contacts and epsilon nodes
    plot.segment(
        x0='x0', y0='y0', x1='x1', y1='y1',
        source=contact_connections_source,
        line_color='#800080', line_width=1.5, alpha=0.8
    )
    
    # Info panel with cleaner CSS for spacing
    info_div = Div(
        text=f"""
        <div style="display: flex; flex-direction: column; gap: 12px;">
            <div>
                <h3 style="margin-bottom: 10px;">Simulation Details</h3>
                <b>Deployment Area:</b> {AREA_LENGTH}km × {AREA_WIDTH}km<br>
                <b>Distance from Shore:</b> {SHORE_DISTANCE}km to {SHORE_DISTANCE + AREA_WIDTH}km<br>
                <b>Sensor Resolution:</b> {simulation.ocean_area.resolution}km<br>
                <b>Nodes:</b> {len(simulation.epsilon_nodes)} ε-nodes (blue), {len(simulation.sigma_nodes)} σ-nodes (red)<br>
                <b>Marine Entities:</b> {len(simulation.theta_contacts)} θ-contacts<br>
                <b>Status:</b> Ready
            </div>
            
            <div>
                <h4 style="margin-bottom: 8px;">Entity Types:</h4>
                <ul style="padding-left: 20px; margin-top: 0;">
                  <li>Detected θ-contacts: <span style="color: #00cc00;">●</span> (bright green)</li>
                  <li>Undetected θ-contacts: <span style="color: #999999;">●</span> (grey)</li>
                </ul>
            </div>
            
            <div>
                <h4 style="margin-bottom: 8px;">Marine Species:</h4>
                <div id="contacts-list" style="max-height: 150px; overflow-y: auto; margin-top: 0;">
                </div>
            </div>
        </div>
        """,
        width=400
    )

    motion_checkbox.on_change("active", lambda attr, old, new: simulation_config.set_motion_toggles(0 in new))

    
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

    # Import Spacer for better layout management
    from bokeh.models import Spacer
    
    # Control headers with proper styling
    control_header = Div(
        text="<h4 style='margin-bottom: 12px;'>Simulation Controls:</h4>",
        margin=(0, 0, 0, 0)
    )
    
    speed_header = Div(
        text="<h4 style='margin-bottom: 12px;'>Simulation Speed:</h4>",
        margin=(0, 0, 0, 0)
    )
    
    # Group the control buttons with proper spacing
    simulation_buttons = row(
        start_button, 
        stop_button, 
        spacing=2
    )
    
    # Create the speed button rows with proper spacing
    speed_buttons_row1 = row(
        speed_realtime, speed_x4, speed_x8, speed_x16, speed_x32,
        spacing=1
    )
    
    speed_buttons_row2 = row(
        speed_x64, speed_x128, speed_x256,
        spacing=1
    )
    
    # Organize the sections using proper vertical layout with spacers
    controls_section = column(
        control_header,
        simulation_buttons,
        sizing_mode="stretch_width",
        spacing=1
    )
    
    speed_section = column(
        speed_header,
        speed_buttons_row1,
        Spacer(height=5),
        speed_buttons_row2,
        sizing_mode="stretch_width",
        spacing=1
    )
    
    # Create the main control column with proper layout and spacing
    controls = column(
        info_div,
        Spacer(height=5),
        motion_checkbox,
        Spacer(height=5),
        controls_section,
        Spacer(height=5),
        speed_section,
        sizing_mode="stretch_width",
        spacing=5
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
        
        # Update connections between ε-nodes
        permanent_connections_data = {
            'x0': [], 'y0': [], 'x1': [], 'y1': [], 'id0': [], 'id1': []
        }
        intermittent_connections_data = {
            'x0': [], 'y0': [], 'x1': [], 'y1': [], 'id0': [], 'id1': []
        }
        
        # Create a dictionary to map node IDs to positions for faster lookup
        epsilon_positions = {node.id: node.position for node in simulation.epsilon_nodes}
        
        # Process permanent connections
        for id1, id2 in simulation.epsilon_connections['permanent']:
            if id1 in epsilon_positions and id2 in epsilon_positions:
                pos1 = epsilon_positions[id1]
                pos2 = epsilon_positions[id2]
                permanent_connections_data['x0'].append(pos1[0])
                permanent_connections_data['y0'].append(pos1[1])
                permanent_connections_data['x1'].append(pos2[0])
                permanent_connections_data['y1'].append(pos2[1])
                permanent_connections_data['id0'].append(id1)
                permanent_connections_data['id1'].append(id2)
        
        # Process intermittent connections
        for id1, id2 in simulation.epsilon_connections['intermittent']:
            if id1 in epsilon_positions and id2 in epsilon_positions:
                pos1 = epsilon_positions[id1]
                pos2 = epsilon_positions[id2]
                intermittent_connections_data['x0'].append(pos1[0])
                intermittent_connections_data['y0'].append(pos1[1])
                intermittent_connections_data['x1'].append(pos2[0])
                intermittent_connections_data['y1'].append(pos2[1])
                intermittent_connections_data['id0'].append(id1)
                intermittent_connections_data['id1'].append(id2)
        
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
        permanent_connections_source.data = permanent_connections_data
        intermittent_connections_source.data = intermittent_connections_data
        
        # Update connections between detected contacts and epsilon nodes
        contact_connections_data = {
            'x0': [], 'y0': [], 'x1': [], 'y1': []
        }

        for contact in simulation.theta_contacts:
            for node in simulation.epsilon_nodes:
                distance = np.linalg.norm(node.position - contact.position)
                if distance <= detection_radius_km:
                    contact_connections_data['x0'].append(node.position[0])
                    contact_connections_data['y0'].append(node.position[1])
                    contact_connections_data['x1'].append(contact.position[0])
                    contact_connections_data['y1'].append(contact.position[1])

        # Update contact connections source
        contact_connections_source.data = contact_connections_data

        # Add contact connections (purple lines)
        plot.segment(
            x0='x0', y0='y0', x1='x1', y1='y1',
            source=contact_connections_source,
            line_color='#800080', line_width=1.5, alpha=0.8
        )
    
        # Update current direction indicator
        if len(simulation.epsilon_nodes) > 0:
            # Get a sample current vector from the center of the area
            center_pos = [AREA_LENGTH/2, SHORE_DISTANCE + AREA_WIDTH/2]
            current_vector = simulation.ocean_area.calculate_ocean_current(center_pos, simulation.current_time)
            
            # Scale for visualization
            scale_factor = 10.0  # Adjust for visibility
            dx, dy = current_vector
            strength = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
            
            # Update arrow vector
            current_indicator_source.data['x_end'] = [current_indicator_source.data['x'][0] + dx * scale_factor]
            current_indicator_source.data['y_end'] = [current_indicator_source.data['y'][0] + dy * scale_factor]
            
            # Update arrow layout with new size
            arrow = Arrow(end=VeeHead(size=arrow_head_size * 2, fill_color="#0066cc", line_color="#0066cc"),
                         x_start='x', y_start='y', x_end='x_end', y_end='y_end',
                         source=current_indicator_source,
                         line_width=2, line_color="#0066cc")
            plot.add_layout(arrow)
            
            # Update text
            direction_name = get_direction_name(direction)
            current_info.text = f"Current: {direction_name} ({strength*100:.1f} cm/s)"
        
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
        
        # Build contacts table HTML
        contacts_table_html = """
        <table style='border-collapse: collapse; width: 100%;'>
            <thead>
                <tr>
                    <th style='border: 1px solid #ddd; padding: 8px;'>ID</th>
                    <th style='border: 1px solid #ddd; padding: 8px; width: 30%;'>English Name</th>
                    <th style='border: 1px solid #ddd; padding: 8px; width: 30%;'>Latin Name</th> <!-- Increased width -->
                    <th style='border: 1px solid #ddd; padding: 8px; width: 30%;'>Status</th> <!-- Increased width -->
                </tr>
            </thead>
            <tbody>
        """

        # Sort contacts by ID to ensure consistent ordering
        sorted_contacts = sorted(simulation.theta_contacts, key=lambda x: x.id)
        
        for contact in sorted_contacts:
            # Check if detected
            is_detected = any(contact.id == id for id in detected_data['id'])
            status_color = "#00cc00" if is_detected else "#999999"
            status_text = "Detected" if is_detected else "Not detected"

            # Extract English and Latin names
            english_name = contact.type.replace('_', ' ').title()
            latin_name = contact.species_name

            contacts_table_html += f"""
            <tr>
                <td style='border: 1px solid #ddd; padding: 8px;'>{contact.id}</td>
                <td style='border: 1px solid #ddd; padding: 8px;'>{english_name}</td>
                <td style='border: 1px solid #ddd; padding: 8px; width: 30%;'>{latin_name}</td>
                <td style='border: 1px solid #ddd; padding: 8px; width: 20%; color: {status_color};'>{status_text}</td>
            </tr>
            """

        contacts_table_html += """
            </tbody>
        </table>
        """

        # Replace the old contacts list with the new table
        info_text = f"""
        <div style="display: flex; flex-direction: column; gap: 12px;">
            <div>
                <h3 style="margin-bottom: 10px;">Simulation Details</h3>
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
            </div>
            <div>
                <h3 style="margin-bottom: 10px;">Marine Entities</h3>
                {contacts_table_html}
            </div>
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


# -----------------------------------------------------------------------------
# CLI – combines both modes with OmegaConf support
# -----------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Aquascan Marine Simulation")

    # Config file
    parser.add_argument('--cfg', default='configs/base.yml', 
                       help='Configuration file path')
    
    # Mode flags
    parser.add_argument("--headless", action="store_true",
                        help="Run without Bokeh (CI / batch)")

    # Common params
    parser.add_argument("--ticks", type=int,
                        help="Number of ticks to simulate (headless only)")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    parser.add_argument("--out", type=str,
                        help="Output path for HDF5 snapshot file (headless only)")

    # Visual‑only params
    parser.add_argument("--port", type=int,
                        help="Port for Bokeh server (visual mode)")
    parser.add_argument("--show", action="store_true",
                        help="Open browser automatically (visual mode)")

    args = parser.parse_args(argv)
    
    # Load configuration from file
    try:
        cfg = OmegaConf.load(args.cfg)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        cfg = OmegaConf.create({
            'sim': {'ticks': 600, 'seed': 42},
            'bokeh': {'port': 5006, 'show': False}
        })
    
    # Override with command line arguments
    if args.ticks is not None:
        cfg.sim.ticks = args.ticks
    if args.seed is not None:
        cfg.sim.seed = args.seed
    if args.port is not None:
        cfg.bokeh.port = args.port
    if args.show:
        cfg.bokeh.show = True
    
    # Use headless mode based on command line flag
    visual = not args.headless

    if not visual:
        # ------------------------------------------------------------------
        # Headless fast‑path – returns immediately when done
        # ------------------------------------------------------------------
        print(f"[Aquascan] Headless run: ticks={cfg.sim.ticks}, seed={cfg.sim.seed}")
        start = time.time()
        
        # Process output path if provided
        out_path = None
        if args.out:
            out_path = Path(args.out)
            # Create directory if it doesn't exist
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Aquascan] Will export snapshot to {out_path}")
        
        # Run simulation
        run(ticks=cfg.sim.ticks, seed=cfg.sim.seed, visual=False, out_path=out_path)
        
        print(f"[Aquascan] Completed in {time.time() - start:.2f}s")
        return

    # ----------------------------------------------------------------------
    # Interactive Bokeh server (legacy behaviour)
    # ----------------------------------------------------------------------
    print(f"[Aquascan] Starting Bokeh server on port {cfg.bokeh.port}")
    print("Press Ctrl+C to stop")

    server = Server({"/": bokeh_app}, port=cfg.bokeh.port, io_loop=IOLoop.current(),
                   allow_websocket_origin=[f"localhost:{cfg.bokeh.port}"])
    server.start()

    if cfg.bokeh.show:
        server.io_loop.add_callback(server.show, "/")

    server.io_loop.start()


if __name__ == "__main__":
    main()
