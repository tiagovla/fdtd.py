"""This module implement detectors."""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .bases import FDTDElementBase
from .utils import Direction

logger = logging.getLogger(__name__)


class Detector(FDTDElementBase):
    """Base class for all detectors.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the detector.
    y_min : float
        Minimum y coordinate of the bounding box containing the detector.
    z_min : float
        Minimum z coordinate of the bounding box containing the detector.
    x_max : float
        Maximum x coordinate of the bounding box containing the detector.
    y_max : float
        Maximum y coordinate of the bounding box containing the detector.
    z_max : float
        Maximum z coordinate of the bounding box containing the detector.
    name : Optional[str]
        Name of the source.
    plot : bool
        Flag to enable plotting.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
        plot: bool = False,
    ):
        """Initialize the object."""
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.name = name
        self.direction = direction
        self.captured: Optional[np.ndarray] = None
        self._plot = plot

        self.i_s: Optional[slice] = None
        self.j_s: Optional[slice] = None
        self.k_s: Optional[slice] = None
        self.idx_s: Optional[Tuple] = None
        self.idx_e: Optional[Tuple] = None

    def update(self):
        """Update detectors."""

    def attach_to_grid(self):
        """Attach object to grid."""

    def plot_3d(self, ax, alpha=0.5):
        """Plot a source and attach to an axis."""

    def plot(self):
        """Plot."""


class VoltageDetector(Detector):
    """Model of a voltage detector.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the detector.
    y_min : float
        Minimum y coordinate of the bounding box containing the detector.
    z_min : float
        Minimum z coordinate of the bounding box containing the detector.
    x_max : float
        Maximum x coordinate of the bounding box containing the detector.
    y_max : float
        Maximum y coordinate of the bounding box containing the detector.
    z_max : float
        Maximum z coordinate of the bounding box containing the detector.
    name : Optional[str]
        Name of the source.
    plot : bool
        Flag to enable plotting.
    """

    def attach_to_grid(self):
        """Attach object to grid."""
        s = self.idx_s = (
            np.argmin(np.abs(self.grid.x - self.x_min)),
            np.argmin(np.abs(self.grid.y - self.y_min)),
            np.argmin(np.abs(self.grid.z - self.z_min)),
        )
        e = self.idx_e = (
            np.argmin(np.abs(self.grid.x - self.x_max)),
            np.argmin(np.abs(self.grid.y - self.y_max)),
            np.argmin(np.abs(self.grid.z - self.z_max)),
        )

        dx, dy, dz = self.grid.spacing

        if self.direction == Direction.X:
            self.i_s = slice(self.idx_s[0], self.idx_e[0])
            self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
            self.c_v = -dx / ((e[1] - s[1] + 1) * (e[2] - s[2] + 1))

        elif self.direction == Direction.Y:
            self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            self.j_s = slice(self.idx_s[1], self.idx_e[1])
            self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
            self.c_v = -dy / ((e[2] - s[2] + 1) * (e[0] - s[0] + 1))

        else:
            self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.k_s = slice(self.idx_s[2], self.idx_e[2])
            self.c_v = -dz / ((e[0] - s[0] + 1) * (e[1] - s[1] + 1))

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#B0BF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])

    def plot(self):
        """Plot."""
        if self._plot:
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(self.captured)
            plt.show()

    def update(self):
        """Capture voltage."""
        try:
            self.captured[self.grid.current_time_step] = self.c_v * np.sum(
                self.grid.E[self.i_s, self.j_s, self.k_s,
                            self.direction.value])
        except TypeError:
            self.captured = np.zeros(self.grid.n_steps)
            self.update()


class HFieldDetector(Detector):
    """Model of a magnetic field detector.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the detector.
    y_min : float
        Minimum y coordinate of the bounding box containing the detector.
    z_min : float
        Minimum z coordinate of the bounding box containing the detector.
    x_max : float
        Maximum x coordinate of the bounding box containing the detector.
    y_max : float
        Maximum y coordinate of the bounding box containing the detector.
    z_max : float
        Maximum z coordinate of the bounding box containing the detector.
    name : Optional[str]
        Name of the source.
    plot : bool
        Flag to enable plotting.
    """

    def attach_to_grid(self):
        """Attach object to grid."""
        s = self.idx_s = (
            np.argmin(np.abs(self.grid.x - self.x_min)),
            np.argmin(np.abs(self.grid.y - self.y_min)),
            np.argmin(np.abs(self.grid.z - self.z_min)),
        )
        e = self.idx_e = (
            np.argmin(np.abs(self.grid.x - self.x_max)),
            np.argmin(np.abs(self.grid.y - self.y_max)),
            np.argmin(np.abs(self.grid.z - self.z_max)),
        )
        self.size = (e[0] - s[0] + 1, e[1] - s[1] + 1, e[2] - s[2] + 1)

        self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
        self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
        self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
        logger.debug(
            f"EFieldDetector attached to {self.i_s}, {self.j_s} and {self.k_s}"
        )

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#B0BF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])

    def plot(self):
        """Plot."""
        if not self._plot:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        logger.debug(f"shape of captured {self.captured.shape}")
        ax.plot(self.captured.flatten())
        plt.show()

    def pos_processing(self):
        """Pos processing of results."""
        self.values_time = self.captured.flatten()
        self.time = self.grid.dt * np.arange(0, self.grid.n_steps)

        # FFT:
        N = len(self.values_time)
        self.values_freq = np.fft.fft(self.values_time)[:N // 2] / N
        self.freq = np.fft.fftfreq(N, self.grid.dt)[:N // 2]

    def update(self):
        """Capture H Field."""
        try:
            self.captured[:, :, :, self.grid.current_time_step] = self.grid.H[
                self.i_s, self.j_s, self.k_s, self.direction.value]
        except TypeError:
            size = self.size + (self.grid.n_steps, )
            self.captured = np.zeros_like(self.grid.H, shape=size)
            self.update()


class EFieldDetector(Detector):
    """Model of an electric field detector.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the detector.
    y_min : float
        Minimum y coordinate of the bounding box containing the detector.
    z_min : float
        Minimum z coordinate of the bounding box containing the detector.
    x_max : float
        Maximum x coordinate of the bounding box containing the detector.
    y_max : float
        Maximum y coordinate of the bounding box containing the detector.
    z_max : float
        Maximum z coordinate of the bounding box containing the detector.
    name : Optional[str]
        Name of the source.
    plot : bool
        Flag to enable plotting.
    """

    def attach_to_grid(self):
        """Attach object to grid."""
        s = self.idx_s = (
            np.argmin(np.abs(self.grid.x - self.x_min)),
            np.argmin(np.abs(self.grid.y - self.y_min)),
            np.argmin(np.abs(self.grid.z - self.z_min)),
        )
        e = self.idx_e = (
            np.argmin(np.abs(self.grid.x - self.x_max)),
            np.argmin(np.abs(self.grid.y - self.y_max)),
            np.argmin(np.abs(self.grid.z - self.z_max)),
        )
        self.size = (e[0] - s[0] + 1, e[1] - s[1] + 1, e[2] - s[2] + 1)

        self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
        self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
        self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
        logger.debug(
            f"EFieldDetector attached to {self.i_s}, {self.j_s} and {self.k_s}"
        )

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#B0BF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])

    def plot(self):
        """Plot."""
        if not self._plot:
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        logger.debug(f"shape of captured {self.captured.shape}")
        ax.plot(self.captured.flatten())
        plt.show()

    def pos_processing(self):
        """Pos processing of results."""
        self.values_time = self.captured.flatten()
        self.time = self.grid.dt * np.arange(0, self.grid.n_steps)

        N = len(self.values_time)
        self.values_freq = np.fft.fft(self.values_time)[:N // 2] / N
        self.freq = np.fft.fftfreq(N, self.grid.dt)[:N // 2]

    def update(self):
        """Capture E Field."""
        try:
            self.captured[:, :, :, self.grid.current_time_step] = self.grid.E[
                self.i_s, self.j_s, self.k_s, self.direction.value]
        except TypeError:
            size = self.size + (self.grid.n_steps, )
            self.captured = np.zeros_like(self.grid.E, shape=size)
            self.update()
