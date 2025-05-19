"""
Simplified Bokeh Visualization Module for debugging

This module provides a simpler visualization approach to help
diagnose rendering issues.
"""

from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Div
import numpy as np
import time

from config.simulation_config import AREA_LENGTH, AREA_WIDTH
from simulation.simulation_loop import AquascanSimulation


def create_simple_visualization():
    """Create a simple visualization for debugging."""
    # Create simulation instance
    simulation = AquascanSimulation()
    simulation.initialize()
    
    # Print node information for debugging
    print(f"Created simulation with {len(simulation.epsilon_nodes)} ε-nodes, "
          f"{len(simulation.sigma_nodes)} σ-nodes, and {len(simulation.theta_contacts)} θ-contacts")
    
    # Create data sources
    epsilon_data = {
        'x': [node.position[0] for node in simulation.epsilon_nodes],
        'y': [node.position[1] for node in simulation.epsilon_nodes],
    }
    sigma_data = {
        'x': [node.position[0] for node in simulation.sigma_nodes],
        'y': [node.position[1] for node in simulation.sigma_nodes],
    }
    contact_data = {
        'x': [contact.position[0] for contact in simulation.theta_contacts],
        'y': [contact.position[1] for contact in simulation.theta_contacts],
    }
    
    # Print sample data
    print(f"First 5 ε-nodes: {list(zip(epsilon_data['x'][:5], epsilon_data['y'][:5]))}")
    print(f"All σ-nodes: {list(zip(sigma_data['x'], sigma_data['y']))}")
    print(f"All θ-contacts: {list(zip(contact_data['x'], contact_data['y']))}")
    
    # Create sources
    epsilon_source = ColumnDataSource(epsilon_data)
    sigma_source = ColumnDataSource(sigma_data)
    contact_source = ColumnDataSource(contact_data)
    
    # Create plot
    plot = figure(
        width=800, height=600,
        title="Aquascan Simulation - Debug View",
        x_range=(-1, AREA_LENGTH + 1),
        y_range=(-1, AREA_WIDTH + 1),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    
    # Add simple circle glyphs - no markers, just basic circles
    # Make them very large and with different colors for visibility
    plot.circle(x='x', y='y', source=epsilon_source, size=15, color='blue', alpha=0.7, legend_label="ε-nodes")
    plot.circle(x='x', y='y', source=sigma_source, size=20, color='red', alpha=0.9, legend_label="σ-nodes")
    plot.circle(x='x', y='y', source=contact_source, size=15, color='green', alpha=0.8, legend_label="θ-contacts")
    
    # Add a clearly visible grid
    plot.grid.grid_line_color = "black"
    plot.grid.grid_line_alpha = 0.5
    plot.grid.grid_line_width = 1
    
    # Make sure the background is distinct
    plot.background_fill_color = "lightgray"
    
    # Add axes labels
    plot.xaxis.axis_label = "X (km)"
    plot.yaxis.axis_label = "Y (km)"
    
    # Configure legend
    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"
    
    # Info display
    info_div = Div(
        text=f"<b>Aquascan Debug View</b><br>{len(epsilon_data['x'])} ε-nodes, "
             f"{len(sigma_data['x'])} σ-nodes, {len(contact_data['x'])} θ-contacts",
        width=400
    )
    
    # Start button
    start_button = Button(label="Start Simulation", button_type="success")
    
    # Create layout
    layout = column(
        info_div,
        start_button,
        plot
    )
    
    # Add to document
    doc = curdoc()
    doc.add_root(layout)
    doc.title = "Aquascan Debug Visualization"
    
    # Add a callback to update the title when the button is clicked
    def on_button_click():
        plot.title.text = "Simulation Started (Debug View)"
        info_div.text = f"<b>Simulation Running</b><br>{len(epsilon_data['x'])} ε-nodes, " \
                        f"{len(sigma_data['x'])} σ-nodes, {len(contact_data['x'])} θ-contacts"
    
    start_button.on_click(on_button_click)
    
    return layout


def initialize():
    """Initialize the simplified Bokeh app."""
    return create_simple_visualization()
