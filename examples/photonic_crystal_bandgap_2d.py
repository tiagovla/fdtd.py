import copy
import logging
import random
from logging.config import fileConfig

import matplotlib.pyplot as plt
import numpy as np
from morpho import BrillouinZonePath as BZPath
from morpho import SymmetryPoint as SPoint
from scipy.signal import find_peaks

from fdtd import EFieldDetector, Grid, HFieldDetector, Material
from fdtd.boundaries import PeriodicBlochBoundary
from fdtd.constants import SPEED_LIGHT as C
from fdtd.objects import Sphere
from fdtd.sources import ImpressedElectricCurrentSource as JSource
from fdtd.sources import ImpressedMagneticCurrentSource as MSource

# Setting up logger
# fileConfig("logging_config.ini")
logger = logging.getLogger("fdtd")

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid_tm = Grid(shape=(64, 64, 1), spacing=(1/64) * 1e-6)
grid_te = Grid(shape=(64, 64, 1), spacing=(1/64) * 1e-6)

print("Creating components...")

j_sources = []
e_detectors = []

j_sources = []
for _ in range(5):
    point = (random.uniform(0.1e-6, 0.9e-6), random.uniform(0.1e-6, 0.9e-6), 0)
    j_source = JSource(*point, *point)
    j_sources.append(j_source)

e_detectors = []
for _ in range(5):
    point = (random.uniform(0.1e-6, 0.9e-6), random.uniform(0.1e-6, 0.9e-6), 0)
    e_detector = EFieldDetector(*point, *point)
    e_detectors.append(e_detector)

m_sources = []
for _ in range(5):
    point = (random.uniform(0.1e-6, 0.9e-6), random.uniform(0.1e-6, 0.9e-6), 0)
    m_source = MSource(*point, *point)
    m_sources.append(m_source)

h_detectors = []
for _ in range(5):
    point = (random.uniform(0.1e-6, 0.9e-6), random.uniform(0.1e-6, 0.9e-6), 0)
    h_detector = HFieldDetector(*point, *point)
    h_detectors.append(h_detector)

p_boundary_tm = PeriodicBlochBoundary(x_direction=True, y_direction=True)

p_boundary_te = copy.deepcopy(p_boundary_tm)

diel_mat = Material(name="diel", eps_r=8.9)

sphere_tm = Sphere(*(0.5e-6, 0.5e-6, 0), radius=0.2e-6, material=diel_mat)
sphere_te = copy.deepcopy(sphere_tm)

print("Adding components to grid...")

grid_tm.add(sphere_tm)
grid_tm.add(p_boundary_tm)
for j_source in j_sources:
    grid_tm.add(j_source)
for e_detector in e_detectors:
    grid_tm.add(e_detector)

grid_te.add(sphere_te)
grid_te.add(p_boundary_te)
for m_source in m_sources:
    grid_te.add(m_source)
for h_detector in h_detectors:
    grid_te.add(h_detector)

print("Running simulation...")

n_steps = 20000
frames = 10
a = 1e-6

G = SPoint((0, 0), "Γ")
X = SPoint((1 / 2, 0), "X")
M = SPoint((1 / 2, 1 / 2, 0), "M")
t1, t2 = (a, 0), (0, a)
bz_path = BZPath([G, X, M, G], t1, t2, n_points=50)

betas = [bz_path.beta_vec[:, col] for col in range(bz_path.beta_vec.shape[1])]
betas_len = bz_path.beta_vec_len

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xticklabels(bz_path.symmetry_names)
ax.set_xticks(bz_path.symmetry_locations)
ax.set_xlim(0, bz_path.symmetry_locations[-1])
ax.set_ylim(0, 0.8)
ax.set_xlabel(r"Bloch Wave Vector $\beta$")
ax.set_ylabel(r"Frequency $\omega a/2\pi c}$")
ax.grid(True)
fig.tight_layout()
plt.ion()
plt.show()

for beta, beta_len in zip(betas, betas_len):
    p_boundary_tm.b_vec = (beta[0], beta[1], 0)
    grid_tm.reset()
    grid_tm.run(n_steps=n_steps)

    print("Showing results...")
    psd = np.zeros((n_steps // 2, ))
    for e_detector in e_detectors:
        e_detector.pos_processing()
        psd += np.abs(e_detector.values_freq)**2

    peaks, _ = find_peaks(np.abs(psd), threshold=1e-30)
    fs = e_detectors[0].freq[peaks]

    ax.scatter(beta_len * (1 + 0*fs),
               fs * a / C,
               color="b",
               marker=".",
               label="TM")

    plt.draw()
    plt.pause(0.001)

for beta, beta_len in zip(betas, betas_len):
    p_boundary_te.b_vec = (beta[0], beta[1], 0)
    grid_te.reset()
    grid_te.run(n_steps=n_steps)

    print("Showing results...")
    psd = np.zeros((n_steps // 2, ))
    for h_detector in h_detectors:
        h_detector.pos_processing()
        psd += np.abs(h_detector.values_freq)**2

    peaks, _ = find_peaks(np.abs(psd), threshold=1e-30)
    fs = h_detectors[0].freq[peaks]

    ax.scatter(beta_len * (1 + 0*fs),
               fs * a / C,
               color="r",
               marker=".",
               label="TE")

    plt.draw()
    plt.pause(0.001)

handles, labels = ax.get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc="best")

plt.ioff()
plt.show()
