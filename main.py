"""This is the main application."""
from fdtd.grid import Grid
from fdtd.materials import Material
from fdtd.objects import Brick, Sphere

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid((10, 10, 10), 0.1)

print("Creating objects...")
brick = Brick(0.44, 0.55, 0.6, 0.5, 0.6, 0.8, "DIEL1")
sphere1 = Sphere(0.3, 0.3, 0.6, 0.1, "DIEL2")
sphere2 = Sphere(0.3, 0.3, 0.6, 0.25, "DIEL1")
grid.add_object(brick)
grid.add_object(sphere1)
grid.add_object(sphere2)

grid.plot_3d()
