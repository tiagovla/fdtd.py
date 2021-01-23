"""This is the main application."""

import logging
from logging.config import fileConfig

from fdtd import Brick, Grid, Material, Resistor, Sphere, VoltageSource

# Setting up logger
fileConfig("logging_config.ini")
logger = logging.getLogger("fdtd")

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid(shape=(10, 10, 10), grid_spacing=0.1)

print("Creating objects...")
brick = Brick(0.1, 0.1, 0.1, 0.8, 0.9, 0.8, "DIEL1")
sphere = Sphere(0.5, 0.5, 0.6, 0.3, "DIEL2")
v_source = VoltageSource(0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 50)
resistor = Resistor(0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 40)

print("Adding components to grid...")
grid.add(brick)
grid.add(sphere)
grid.add(v_source)
grid.add(resistor)

print("Run simulation...")
grid.run(n_steps=500)
grid.plot_disc()
