"""Implements a 3d bandgap calculation."""
import concurrent.futures
import copy
import itertools
import logging
import os
import random
from logging.config import fileConfig

import matplotlib.pyplot as plt
import numpy as np
from morpho import BrillouinZonePath as BZPath
from morpho import SymmetryPoint as SPoint
from scipy.signal import find_peaks

from fdtd import EFieldDetector, Grid, Material
from fdtd.boundaries import PeriodicBlochBoundary as PBBoundary
from fdtd.constants import SPEED_LIGHT as C
from fdtd.objects import Brick
from fdtd.sources import ImpressedElectricCurrentSource as JSource
from fdtd.utils import Direction

# Setting up logger
# fileConfig("logging_config.ini")
logger = logging.getLogger("fdtd")

# geometry
a = 1e-6
w = 0.2 * a
n_steps = 20000
n_points = 50

# Define the symmetry points
G = SPoint((0, 0, 0), "Γ")
Z = SPoint((0, 0, 1 / 2), "Z")
X = SPoint((1 / 2, 0, 0), "X")
D = SPoint((1 / 2, 1 / 2, 1 / 2), "D")
T = SPoint((1 / 2, 0, 1 / 2), "T")

t1, t2, t3 = (a, 0, 0), (0, a, 0), (0, 0, a)

# Construct the bloch wave path
bz_path = BZPath([D, Z, G, Z, T, X], t1, t2, t3, n_points=n_points)

print("Loading materials...")

Material.load("materials.json")

print("Creating grid...")

grid = Grid(shape=(64, 64, 64), spacing=(1/64) * 1e-6, courant_factor=0.5)

print("Creating components...")

j_sources = []
e_detectors = []

dir_gen = itertools.cycle([Direction.X, Direction.Y, Direction.Z])
for _ in range(30):
    point = (
        random.uniform(0.2e-6, 0.8e-6),
        random.uniform(0.2e-6, 0.8e-6),
        random.uniform(0.2e-6, 0.8e-6),
    )
    j_source = JSource(*point, *point, direction=next(dir_gen))
    j_sources.append(j_source)

for _ in range(30):
    point = (
        random.uniform(0.2e-6, 0.8e-6),
        random.uniform(0.2e-6, 0.8e-6),
        random.uniform(0.2e-6, 0.8e-6),
    )
    e_detector = EFieldDetector(*point, *point, direction=next(dir_gen))
    e_detectors.append(e_detector)

diel = Material(name="diel", eps_r=2.43)

b1 = Brick(*(0, 0, 0), *(0.5 * w, 0.5 * w, a), material=diel)
b2 = Brick(*(0, a - 0.5*w, 0), *(0.5 * w, a, a), material=diel)
b3 = Brick(*(a - 0.5*w, 0, 0), *(a, 0.5 * w, a), material=diel)
b4 = Brick(*(a - 0.5*w, a - 0.5*w, 0), *(a, a, a), material=diel)
b5 = Brick(*(0, 0, 0), *(a, 0.5 * w, 0.5 * w), material=diel)
b6 = Brick(*(0, 0, a - 0.5*w), *(a, 0.5 * w, a), material=diel)
b7 = Brick(*(0, a - 0.5*w, 0), *(a, a, 0.5 * w), material=diel)
b8 = Brick(*(0, a - 0.5*w, a - 0.5*w), *(a, a, a), material=diel)
b9 = Brick(*(0, 0, 0), *(0.5 * w, a, 0.5 * w), material=diel)
b10 = Brick(*(0, 0, a - 0.5*w), *(0.5 * w, a, a), material=diel)
b11 = Brick(*(a - 0.5*w, 0, 0), *(a, a, 0.5 * w), material=diel)
b12 = Brick(*(a - 0.5*w, 0, a - 0.5*w), *(a, a, a), material=diel)

bricks = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12]

p_boundary = PBBoundary(x_direction=True, y_direction=True, z_direction=True)

print("Adding components to grid...")
for brick in bricks:
    grid.add(brick)

for j_source in j_sources:
    grid.add(j_source)

for e_detector in e_detectors:
    grid.add(e_detector)
grid.add(p_boundary)

print("Running simulation...")


def simulation(beta):
    sim_grid = copy.deepcopy(grid)
    if sim_grid.boundaries:
        sim_grid.boundaries[0].b_vec = (beta[0], beta[1], beta[2])
    sim_grid.reset()
    sim_grid.run(n_steps=n_steps)

    psd = np.zeros((n_steps // 2, ))
    for e_detector in sim_grid.detectors:
        e_detector.pos_processing()
        psd += np.abs(e_detector.values_freq)**2

    peaks, _ = find_peaks(np.abs(psd), threshold=1e-25)
    return sim_grid.detectors[0].freq[peaks]


betas = [bz_path.beta_vec[:, col] for col in range(bz_path.beta_vec.shape[1])]
betas_len = bz_path.beta_vec_len

fig, ax = plt.subplots(figsize=(5, 4))
plt.style.use("seaborn-ticks")
ax.set_xticklabels(bz_path.symmetry_names)
ax.set_xticks(bz_path.symmetry_locations)
ax.set_xlim(0, bz_path.symmetry_locations[-1])
ax.set_ylim(0, 1.6)
ax.set_xlabel(r"Bloch Wave Vector $\beta$")
ax.set_ylabel(r"Frequency ${\omega a}/{2\pi c}$")
plt.tight_layout()
ax.grid(True)

with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count()) as executor:
    for beta_len, fs in zip(betas_len, executor.map(simulation, betas)):
        print(beta_len, fs)
        ax.scatter(beta_len * (1 + 0*fs), fs * a / C)
        fig.savefig("result.png")
        fig.savefig("result.svg")

plt.show()
