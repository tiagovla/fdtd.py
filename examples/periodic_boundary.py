import logging
from logging.config import fileConfig
from pathlib import Path

from fdtd import EFieldDetector, Grid, Material
from fdtd.boundaries import PeriodicBoundary
from fdtd.sources import ImpressedElectricCurrentSource as JSource
from fdtd.waveforms import GaussianWaveform

# Setting up logger
fileConfig(Path(__file__).with_name("logging_config.ini"))
logger = logging.getLogger("fdtd")

print("Loading materials...")
Material.load(Path(__file__).with_name("materials.json"))

print("Creating grid...")
grid = Grid(shape=(600, 600, 1), spacing=1e-3)

print("Creating components...")
j_source = JSource(*(40e-3, 40e-3, 0e-3), *(40e-3, 40e-3, 0e-3),
                   waveform=GaussianWaveform(t_0=1e-9, tau=2e-10))
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
