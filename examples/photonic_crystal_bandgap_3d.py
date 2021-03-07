"""Implements a 3d bandgap calculation."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import concurrent.futures
import copy
import itertools
import logging
import random
from logging.config import fileConfig

import matplotlib.pyplot as plt
import numpy as np
from fdtd import EFieldDetector, Grid, Material
from fdtd.boundaries import PeriodicBlochBoundary as PBBoundary
from fdtd.constants import SPEED_LIGHT as C
from fdtd.objects import Brick
from fdtd.sources import ImpressedElectricCurrentSource as JSource
from fdtd.utils import Direction
from morpho import BrillouinZonePath as BZPath
from morpho import SymmetryPoint as SPoint
from scipy.signal import find_peaks

# Setting up logger
# fileConfig("logging_config.ini")
logger = logging.getLogger("fdtd")

# geometry
a = 1e-6
w = 0.2 * a
n_steps = 10000
n_points = 40

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

# print("Running simulation...")


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

    peaks, _ = find_peaks(np.abs(psd), threshold=1e-22)
    return sim_grid.detectors[0].freq[peaks]


betas = [bz_path.beta_vec[:, col] for col in range(bz_path.beta_vec.shape[1])]
betas_len = bz_path.beta_vec_len

fig, ax = plt.subplots()
ax.set_xticklabels(bz_path.symmetry_names)
ax.set_xticks(bz_path.symmetry_locations)
ax.set_xlim(0, bz_path.symmetry_locations[-1])
ax.set_ylim(0, 2)
ax.set_xlabel("Bloch Wave Vector $\\beta$")
ax.set_ylabel("Frequency $\\frac{a \\omega}{2\\pi c}$")
ax.grid(True)

with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count()) as executor:
    for beta_len, fs in zip(betas_len, executor.map(simulation, betas)):
        print(beta_len, fs)
        ax.scatter(beta_len * (1 + 0*fs), fs * a / C)

fig.savefig("result.png")
plt.show()

# idx = 0
# fig, ax = plt.subplots()
# for bx, by, bz in zip(b_x, b_y, b_z):
#     p_boundary.b_vec = (bx, by, bz)
#     grid.reset()
#     grid.run(n_steps=n_steps)

#     print("Showing results...")
#     psd = np.zeros((n_steps // 2, ))
#     for e_detector in e_detectors:
#         e_detector.pos_processing()
#         psd += np.abs(e_detector.values_freq)**2

#     peaks, _ = find_peaks(np.abs(psd), threshold=1e-22)

#     ax.scatter(idx * np.ones(psd[peaks].shape),
#                e_detectors[0].freq[peaks] / 1e12)
#     ax.set_ylim([0, 1000])
#     ax.set_xlim([0, frames])
#     # plt.draw()
#     # plt.pause(0.001)
#     idx += 1

#     # fig, ax = plt.subplots()
#     # ax.plot(e_detectors[0].time, np.abs(e_detectors[0].values_time))

#     # fig, ax = plt.subplots()
#     # ax.plot(e_detectors[0].freq / 1e12, psd)
#     # ax.set_xlim([0, 1200])
#     # ax.scatter(e_detectors[0].freq[peaks] / 1e12, psd[peaks])

#     # plt.show()
#     fig.savefig("result.png")

# #     # X, Y = np.meshgrid(np.linspace(0, 0.5, frames), e_detector.freq / 1e12)
# #     # print("shapes")
# #     # print(X.shape)
# #     # print(history_freq.shape)

# #     # fig, ax = plt.subplots()
# #     # ax.pcolormesh(X, Y, np.abs(history_freq), cmap="RdBu")
# #     # ax.set_ylim([0, 500])
# #     # plt.show()

# #     # fig = plt.figure()
# #     # # ax = fig.add_subplot(111, projection="3d")
# #     # ax = fig.add_subplot(111)
# #     # ax.scatter(X, Y, c=np.abs(history_freq))
# #     # ax.set_ylim([0, 500])

# #     # fig, axs = plt.subplots(3)
# #     # axs[0].plot(e_detector.time, np.abs(e_detector.values_time))
# #     # axs[0].set(xlabel="time (s)", ylabel="|E_z|(V/m)")
# #     # axs[1].plot(e_detector.freq / 1e12, np.abs(e_detector.values_freq))
# #     # axs[1].set(xlabel="frequency (THz)", ylabel="F(E_z)")
# #     # axs[1].set_xlim([0, 500])
# #     # axs[2].plot(e_detector.freq / 1e12, np.angle(e_detector.values_freq))
# #     # axs[2].set(xlabel="frequency (THz)", ylabel="F(E_z)")
# #     # axs[2].set_xlim([0, 500])

# #     # peaks, _ = find_peaks(np.abs(e_detector.values_freq), threshold=1e-10)
# #     # axs[1].scatter(e_detector.freq[peaks] / 1e12,
# #     #                abs(e_detector.values_freq[peaks]))
# #     # print(peaks)
# # plt.ioff()
# # plt.show()

# # # # plot results
# # # # print(e_detector.captured)
