"""This is the main application."""
from fdtd.grid import Grid
from fdtd.materials import Material
from fdtd.objects import Brick, Sphere
from fdtd.sources import VoltageSource

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid(shape=(10, 10, 10), grid_spacing=0.1)

print("Creating objects...")
brick = Brick(0.1, 0.1, 0.1, 0.8, 0.9, 0.8, "DIEL1")
sphere = Sphere(0.5, 0.5, 0.6, 0.3, "DIEL2")
v_source = VoltageSource(0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 50)

print("Adding components to grid...")
grid.add_object(brick)
grid.add_object(sphere)
grid.add_source(v_source)

print("Preparing simulation...")
grid.prepare()
# grid.plot_planes()
