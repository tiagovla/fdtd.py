"""This is the main application."""

from fdtd import (Brick, Grid, Material, Resistor, VoltageDetector,
                  VoltageSource)

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid(shape=(10, 10, 10), spacing=1e-3, courant_factor=0.9)

print("Creating objects...")
brick1 = Brick(*(4e-3, 2e-3, 8e-3), *(6e-3, 8e-3, 8e-3), "PEC")
brick2 = Brick(*(4e-3, 2e-3, 4e-3), *(6e-3, 8e-3, 4e-3), "PEC")
v_source = VoltageSource(*(4e-3, 2e-3, 4e-3), *(6e-3, 2e-3, 8e-3), 50)
resistor = Resistor(*(4e-3, 8e-3, 4e-3), *(6e-3, 8e-3, 8e-3), 50)
v_detector = VoltageDetector(*(4e-3, 5e-3, 4e-3),
                             *(6e-3, 5e-3, 8e-3),
                             plot=True)

print("Adding components to grid...")
grid.add(brick1)
grid.add(brick2)
grid.add(v_source)
# grid.add(resistor)
grid.add(v_detector)

print("Run simulation...")
grid.run(n_steps=1500)
print(grid.dt)
# grid.plot_3d()
