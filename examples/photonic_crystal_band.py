"""This calculates the bandgaps of a photonic crystal."""

import logging
from logging.config import fileConfig

from fdtd import (
    EFieldDetector,
    EFieldSource,
    Grid,
    Material,
    Sphere,
    VoltageDetector,
    VoltageSource,
)

# Setting up logger
fileConfig("logging_config.ini")
logger = logging.getLogger("fdtd")

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid(shape=(500, 500, 0), grid_spacing=1e-3)

print("Creating objects...")
# sphere1 = Sphere(*(10e-3, 10e-3, 0), 3e-3, "DIEL2")
e_source = EFieldSource(*(250e-3, 250e-3, 0e-3), *(250e-3, 250e-3, 0e-3))
e_detector = EFieldDetector(*(95e-3, 95e-3, 0), *(95e-3, 95e-3, 0), plot=True)
# e_detector = EFieldDetector(*(0e-3, 0e-3, 0),
# *(9999e-3, 9999e-3, 0),
# plot=True)

# v_source = VoltageSource(*(30e-3, 30e-3, 4e-3), *(35e-3, 35e-3, 6e-3), 50)
# v_detector = VoltageDetector(*(40e-3, 40e-3, 4e-3),
#                              *(45e-3, 45e-3, 6e-3),
#                              plot=True)
grid.add(e_source)
grid.add(e_detector)
# print("Adding components to grid...")
# grid.add(sphere1)
# grid.add(e_source)

# print("Run simulation...")
grid.run(n_steps=2000)
# print(e_detector.captured.shape)
# print(grid.dt)
# grid.plot_3d(z_view=True)
# grid.plot_disc3()
print(grid.dt)
