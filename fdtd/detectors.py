"""This module implement detectors."""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .bases import FDTDElementBase
from .utils import Direction

logger = logging.getLogger(__name__)


class Detector(FDTDElementBase):
    """An source to be placed in the grid."""

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
        self.captured: np.ndarray = None
        self._plot = plot

    def update(self):
        """Update detectors."""
        pass

    def attach_to_grid(self):
        """Attach object to grid."""
        pass

    def plot_3d(self, ax, alpha=0.5):
        """Plot a source and attach to an axis."""
        pass

    def plot(self):
        """Plot."""
        pass


class VoltageDetector(Detector):
    """Implement a voltage detector."""

    def attach_to_grid(self):
        """Attach object to grid."""
        s = self.idx_s = (
            np.argmin(np.abs(self.grid._x - self.x_min)),
            np.argmin(np.abs(self.grid._y - self.y_min)),
            np.argmin(np.abs(self.grid._z - self.z_min)),
        )
        e = self.idx_e = (
            np.argmin(np.abs(self.grid._x - self.x_max)),
            np.argmin(np.abs(self.grid._y - self.y_max)),
            np.argmin(np.abs(self.grid._z - self.z_max)),
        )

        dx, dy, dz = self.grid.grid_spacing

        if self.direction == Direction.X:
            self.I = slice(self.idx_s[0], self.idx_e[0])
            self.J = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.K = slice(self.idx_s[2], self.idx_e[2] + 1)
            self.c_v = -dx / ((e[1] - s[1] + 1) * (e[2] - s[2] + 1))
            self.index = 0

        elif self.direction == Direction.Y:
            self.I = slice(self.idx_s[0], self.idx_e[0])
            self.J = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.K = slice(self.idx_s[2], self.idx_e[2] + 1)
            self.c_v = -dy / ((e[2] - s[2] + 1) * (e[0] - s[0] + 1))
            self.index = 1

        else:
            self.I = slice(self.idx_s[0], self.idx_e[0] + 1)
            self.J = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.K = slice(self.idx_s[2], self.idx_e[2])
            self.c_v = -dz / ((e[0] - s[0] + 1) * (e[1] - s[1] + 1))
            self.index = 2

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
                self.grid.E[self.I, self.J, self.K, self.index])
        except TypeError:
            self.captured = np.zeros(self.grid.n_steps)
            self.update()
