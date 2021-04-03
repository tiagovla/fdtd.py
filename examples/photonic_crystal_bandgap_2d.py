import copy
import logging
from random import uniform

import matplotlib.pyplot as plt
import numpy as np
from morpho import BrillouinZonePath as BZPath
from morpho import SymmetryPoint as SPoint
from scipy.signal import find_peaks

from fdtd import EFieldDetector, Grid, HFieldDetector, Material
from fdtd.boundaries import PeriodicBlochBoundary as PBBoundary
from fdtd.constants import SPEED_LIGHT as C
from fdtd.objects import Sphere
from fdtd.sources import ImpressedElectricCurrentSource as JSource
from fdtd.sources import ImpressedMagneticCurrentSource as MSource

logger = logging.getLogger("fdtd")

print("Loading materials...")
Material.load("materials.json")

print("Defining initial parameters...")
a = 1e-6  #unit cell size
n = 64  #grid 64x64
r = 0.2 * a  #cylinder radius

print("Creating grid...")
grid_tm = Grid(shape=(n, n, 1), spacing=(1/n) * a)
grid_te = Grid(shape=(n, n, 1), spacing=(1/n) * a)

print("Creating components...")
diel_mat = Material(name="diel", eps_r=8.9)


def rpoints():
    px = uniform(0.1 * a, 0.9 * a)
    py = uniform(0.1 * a, 0.9 * a)
    return (px, py, 0, px, py, 0)


j_sources = [JSource(*rpoints()) for _ in range(5)]
m_sources = [MSource(*rpoints()) for _ in range(5)]
e_detectors = [EFieldDetector(*rpoints()) for _ in range(5)]
h_detectors = [HFieldDetector(*rpoints()) for _ in range(5)]

p_boundary_tm = PBBoundary(x_direction=True, y_direction=True)
p_boundary_te = copy.deepcopy(p_boundary_tm)

sphere_tm = Sphere(*(0.5 * a, 0.5 * a, 0), radius=r, material=diel_mat)
sphere_te = copy.deepcopy(sphere_tm)

print("Adding components to grid...")

for elm in [*j_sources, *e_detectors, sphere_tm, p_boundary_tm]:
    grid_tm.add(elm)

for elm in [*m_sources, *h_detectors, sphere_te, p_boundary_te]:
    grid_te.add(elm)

print("Running simulation...")

n_steps = 10000
frames = 10

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
