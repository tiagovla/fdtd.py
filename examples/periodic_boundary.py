import logging
from logging.config import fileConfig

from fdtd import EFieldDetector, Grid, Material
from fdtd.boundaries import PeriodicBoundary
from fdtd.sources import ImpressedElectricCurrentSource as JSource

# Setting up logger
fileConfig("logging_config.ini")
logger = logging.getLogger("fdtd")

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid(shape=(600, 600, 1), spacing=1e-3)

print("Creating components...")
j_source = JSource(*(40e-3, 40e-3, 0e-3), *(40e-3, 40e-3, 0e-3))
# e_detector = EFieldDetector(*(95e-3, 95e-3, 0), *(95e-3, 95e-3, 0), plot=True)
e_detector = EFieldDetector(*(0, 0, 0), *(99999e-3, 99999e-3, 0), plot=True)
p_boundary = PeriodicBoundary(x_direction=True, y_direction=True)

print("Adding components to grid...")
grid.add(j_source)
grid.add(e_detector)
grid.add(p_boundary)

print("Running simulation...")
grid.run(n_steps=2000)

print("Showing results...")
# plot results
