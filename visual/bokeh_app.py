"""
Bokeh Visualization Module

This module implements real-time visualization for the Aquascan simulation.
Responsibilities:
- Create an interactive Bokeh application
- Visualize ε-nodes, σ-nodes, and θ-contacts
- Display ocean currents as vector fields
- Update visualization in real-time
- Provide controls for simulation parameters
"""

import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Div, Slider, Select
from bokeh.models import Arrow, NormalHead, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap
import time

from config.simulation_config import (
    AREA_LENGTH, AREA_WIDTH, ENTITY_COLORS, TIME_STEP,
    VIZ_UPDATE_INTERVAL, PLOT_WIDTH, PLOT_HEIGHT
)
from simulation.simulation_loop import AquascanSimulation


class AquascanVisualization:
    """
    Visualization controller for the Aquascan simulation.
    """
    
    def __init__(self):
        """Initialize the visualization components."""
        # Create simulation instance
        self.simulation = AquascanSimulation()
        
        # Data sources for Bokeh
        self.epsilon_source = ColumnDataSource({
            'x': [], 'y': [], 'id': [], 'color': []
        })
        self.sigma_source = ColumnDataSource({
            'x': [], 'y': [], 'id': [], 'color': []
        })
        self.contact_source = ColumnDataSource({
            'x': [], 'y': [], 'id': [], 'type': [], 'color': []
        })
        self.current_source = ColumnDataSource({
            'x': [], 'y': [], 'u': [], 'v': [], 'magnitude': []
        })
        
        # Create plots
        self._create_plots()
        
        # Create control panel
        self._create_controls()
        
        # Register with simulation
        self.simulation.register_visualization(self.update_data)
        
        # Initialize visualization
        self.update_data(self.simulation)
    
    def _create_plots(self):
        """Create the main plot and its components."""
        # Main figure
        self.plot = figure(
            width=PLOT_WIDTH, height=PLOT_HEIGHT,
            title="Aquascan Marine Simulation",
            x_range=(0, AREA_LENGTH),
            y_range=(0, AREA_WIDTH),
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Configure plot
        self.plot.grid.grid_line_color = "white"
        self.plot.grid.grid_line_alpha = 0.3
        self.plot.xaxis.axis_label = "Distance along coastline (km)"
        self.plot.yaxis.axis_label = "Distance from shore (km)"
        
        # Add glyph for ε-nodes
        self.epsilon_glyph = self.plot.circle(
            x='x', y='y', size=8, source=self.epsilon_source,
            color='color', alpha=0.7, legend_label="ε-nodes"
        )
        
        # Add glyph for σ-nodes
        self.sigma_glyph = self.plot.square(
            x='x', y='y', size=12, source=self.sigma_source,
            color='color', alpha=0.9, legend_label="σ-nodes"
        )
        
        # Add glyph for θ-contacts
        self.contact_glyph = self.plot.triangle(
            x='x', y='y', size=10, source=self.contact_source,
            color='color', alpha=0.8, legend_label="θ-contacts"
        )
        
        # Add vector field for ocean currents
        color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=1)
        self.current_glyph = self.plot.add_layout(
            Arrow(
                x_start='x', y_start='y',
                x_end='x', y_end='y',
                source=self.current_source,
                end=NormalHead(fill_color="blue", line_color="blue", size=6),
                line_color="blue", line_alpha=0.6, line_width=1
            )
        )
        
        # Add legend configuration
        self.plot.legend.location = "top_left"
        self.plot.legend.click_policy = "hide"
    
    def _create_controls(self):
        """Create control panel for the simulation."""
        # Simulation info display
        self.info_div = Div(
            text="Simulation not started",
            width=400, height=100
        )
        
        # Control buttons
        self.start_button = Button(label="Start Simulation", button_type="success")
        self.start_button.on_click(self.start_simulation)
        
        self.stop_button = Button(label="Stop Simulation", button_type="danger")
        self.stop_button.on_click(self.stop_simulation)
        
        # Time factor slider
        self.time_factor_slider = Slider(
            title="Simulation Speed Factor", value=1.0, start=0.1, end=10.0, step=0.1
        )
        
        # Layout
        controls = column(
            self.info_div,
            row(self.start_button, self.stop_button),
            self.time_factor_slider
        )
        
        # Add to document
        doc = curdoc()
        doc.add_root(row(self.plot, controls))
        doc.title = "Aquascan Marine Simulation"
        
        # Add periodic callback for simulation ticks
        self.callback_id = None
    
    def start_simulation(self):
        """Start the simulation and visualization updates."""
        if not self.simulation.is_running:
            self.simulation.initialize()
            self.simulation.start()
            self.update_data(self.simulation)
            
            # Calculate callback interval based on TIME_STEP and speed factor
            interval = int(TIME_STEP * 1000 / self.time_factor_slider.value)
            interval = max(VIZ_UPDATE_INTERVAL, interval)  # Ensure minimum interval
            
            self.callback_id = curdoc().add_periodic_callback(self.simulation_tick, interval)
            self.start_button.label = "Restart Simulation"
    
    def stop_simulation(self):
        """Stop the simulation and visualization updates."""
        if self.simulation.is_running:
            self.simulation.stop()
            if self.callback_id is not None:
                curdoc().remove_periodic_callback(self.callback_id)
                self.callback_id = None
    
    def simulation_tick(self):
        """Perform a simulation tick and update visualization."""
        if not self.simulation.tick():
            # Simulation has stopped
            if self.callback_id is not None:
                curdoc().remove_periodic_callback(self.callback_id)
                self.callback_id = None
    
    def update_data(self, simulation):
        """
        Update visualization data from simulation state.
        
        Args:
            simulation (AquascanSimulation): The simulation instance
        """
        # Update ε-nodes
        epsilon_data = {
            'x': [], 'y': [], 'id': [], 'color': []
        }
        for node in simulation.epsilon_nodes:
            epsilon_data['x'].append(node.position[0])
            epsilon_data['y'].append(node.position[1])
            epsilon_data['id'].append(node.id)
            epsilon_data['color'].append(ENTITY_COLORS["epsilon_node"])
        
        # Update σ-nodes
        sigma_data = {
            'x': [], 'y': [], 'id': [], 'color': []
        }
        for node in simulation.sigma_nodes:
            sigma_data['x'].append(node.position[0])
            sigma_data['y'].append(node.position[1])
            sigma_data['id'].append(node.id)
            sigma_data['color'].append(ENTITY_COLORS["sigma_node"])
        
        # Update θ-contacts
        contact_data = {
            'x': [], 'y': [], 'id': [], 'type': [], 'color': []
        }
        for contact in simulation.theta_contacts:
            contact_data['x'].append(contact.position[0])
            contact_data['y'].append(contact.position[1])
            contact_data['id'].append(contact.id)
            contact_data['type'].append(contact.type)
            contact_data['color'].append(ENTITY_COLORS[contact.type])
        
        # Update currents (sample at grid points)
        grid_size = 1.0  # 1 km grid for currents
        current_data = {
            'x': [], 'y': [], 'u': [], 'v': [], 'magnitude': []
        }
        
        for x in np.arange(0, AREA_LENGTH, grid_size):
            for y in np.arange(0, AREA_WIDTH, grid_size):
                # Get current at this position
                current_vector = simulation.ocean_area.calculate_ocean_current(
                    (x, y), simulation.current_time
                )
                
                # Scale for visualization
                scale = 0.5  # Adjust to make arrows visible
                dx, dy = current_vector
                
                current_data['x'].append(x)
                current_data['y'].append(y)
                current_data['u'].append(x + dx * scale)
                current_data['v'].append(y + dy * scale)
                current_data['magnitude'].append(np.sqrt(dx**2 + dy**2))
        
        # Update data sources
        self.epsilon_source.data = epsilon_data
        self.sigma_source.data = sigma_data
        self.contact_source.data = contact_data
        self.current_source.data = current_data
        
        # Update info display
        elapsed_time = simulation.current_time
        hours = int(elapsed_time / 3600)
        minutes = int((elapsed_time % 3600) / 60)
        seconds = int(elapsed_time % 60)
        
        status = "running" if simulation.is_running else "stopped"
        
        info_text = f"""
        <b>Simulation {status}</b><br>
        Time: {hours:02d}:{minutes:02d}:{seconds:02d}<br>
        Detections: {simulation.stats['detections']}<br>
        Messages delivered: {simulation.stats['messages_delivered']}
        """
        self.info_div.text = info_text


# Initialize visualization when module is run
def initialize():
    """Initialize the Bokeh visualization app."""
    viz = AquascanVisualization()
    return viz


# Create visualization app
if __name__ == "__main__":
    initialize()
