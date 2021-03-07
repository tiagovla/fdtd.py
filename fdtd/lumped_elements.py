"""This module implement lumped elements."""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import numpy as np

from .bases import FDTDElementBase
from .constants import FREESPACE_PERMITTIVITY as EPS_0
from .utils import BoundingBox, Direction

logger = logging.getLogger(__name__)


class LumpedElement(FDTDElementBase):
    """A lumped element to be placed in the grid."""

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
    ):
        """Initialize the lumped element."""
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.name = name if name else self._create_new_name()
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)
        self.direction = direction

    def attach_to_grid(self):
        """Attach object to grid."""

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#ff0000")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class Resistor(LumpedElement):
    """Implement a resistor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        super().__init__(x_min, y_min, z_min, x_max, y_max, z_max)
        self.resistance = resistance

    def __repr__(self):
        """Dev. string representation."""
        return (f"{self.__class__.__name__}"
                f"[name={self.name}, resistance={self.resistance}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")

    def attach_to_grid(self):
        """Attach object to grid."""
        self.idx_s = (
            np.argmin(np.abs(self.grid._x - self.x_min)),
            np.argmin(np.abs(self.grid._y - self.y_min)),
            np.argmin(np.abs(self.grid._z - self.z_min)),
        )
        self.idx_e = (
            np.argmin(np.abs(self.grid._x - self.x_max)),
            np.argmin(np.abs(self.grid._y - self.y_max)),
            np.argmin(np.abs(self.grid._z - self.z_max)),
        )

        dx, dy, dz = self.grid.grid_spacing
        dt = self.grid.dt
        Rs = self.resistance
        term = (dt*dz) / (Rs*dx*dy)

        if self.direction == Direction.X:
            I = slice(self.idx_s[0], self.idx_e[0])
            J = slice(self.idx_s[1], self.idx_e[1] + 1)
            K = slice(self.idx_s[2], self.idx_e[2] + 1)

            eps = self.grid.eps_r[I, J, K, 0] * EPS_0
            sigma_e = self.grid.sigma_e[I, J, K, 0]

            self.grid.c_ee[I, J, K, 0] = (2*eps - dt*sigma_e -
                                          term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 1] = (2*dt) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 2] = self.grid.c_eh[I, J, K, 1]

        elif self.direction == Direction.Y:
            I = slice(self.idx_s[0], self.idx_e[0])
            J = slice(self.idx_s[1], self.idx_e[1] + 1)
            K = slice(self.idx_s[2], self.idx_e[2] + 1)

            eps = self.grid.eps_r[I, J, K, 1] * EPS_0
            sigma_e = self.grid.sigma_e[I, J, K, 1]

            self.grid.c_ee[I, J, K, 1] = (2*eps - dt*sigma_e -
                                          term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 0] = (2*dt) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 2] = self.grid.c_eh[I, J, K, 1]

        else:
            I = slice(self.idx_s[0], self.idx_e[0] + 1)
            J = slice(self.idx_s[1], self.idx_e[1] + 1)
            K = slice(self.idx_s[2], self.idx_e[2])
            eps = self.grid.eps_r[I, J, K, 2] * EPS_0
            sigma_e = self.grid.sigma_e[I, J, K, 2]

            self.grid.c_ee[I, J, K, 2] = (2*eps - dt*sigma_e -
                                          term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 2] = (2*dt) / (2*eps + dt*sigma_e + term)


class Capacitor(LumpedElement):
    """Implement a capacitor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


class Inductor(LumpedElement):
    """Implement a inductor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


class Diode(LumpedElement):
    """Implement a diode element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
