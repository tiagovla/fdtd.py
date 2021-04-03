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
    """Base class for all lumped elements.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the lumped element.
    y_min : float
        Minimum y coordinate of the bounding box containing the lumped element.
    z_min : float
        Minimum z coordinate of the bounding box containing the lumped element.
    x_max : float
        Maximum x coordinate of the bounding box containing the lumped element.
    y_max : float
        Maximum y coordinate of the bounding box containing the lumped element.
    z_max : float
        Maximum z coordinate of the bounding box containing the lumped element.
    name : Optional[str]
        Name of the lumped element.
    direction : Direction
        Direction of the lumped element.
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
        self.direction = direction

        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)

    def attach_to_grid(self):
        """Attach object to grid."""
        raise NotImplementedError

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
    """Model of a resistor element.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the resistor.
    y_min : float
        Minimum y coordinate of the bounding box containing the resistor.
    z_min : float
        Minimum z coordinate of the bounding box containing the resistor.
    x_max : float
        Maximum x coordinate of the bounding box containing the resistor.
    y_max : float
        Maximum y coordinate of the bounding box containing the resistor.
    z_max : float
        Maximum z coordinate of the bounding box containing the resistor.
    resistance : float
        Internal resistance of the resistor.
    """

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
            np.argmin(np.abs(self.grid.x - self.x_min)),
            np.argmin(np.abs(self.grid.y - self.y_min)),
            np.argmin(np.abs(self.grid.z - self.z_min)),
        )
        self.idx_e = (
            np.argmin(np.abs(self.grid.x - self.x_max)),
            np.argmin(np.abs(self.grid.y - self.y_max)),
            np.argmin(np.abs(self.grid.z - self.z_max)),
        )

        dx, dy, dz = self.grid.spacing
        dt = self.grid.dt
        r_s = self.resistance

        if self.direction == Direction.X:
            i_s = slice(self.idx_s[0], self.idx_e[0])
            j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
            term = (dt*dx) / (r_s*dy*dz)
        elif self.direction == Direction.Y:
            i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            j_s = slice(self.idx_s[1], self.idx_e[1])
            k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
            term = (dt*dy) / (r_s*dx*dz)
        else:
            i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            k_s = slice(self.idx_s[2], self.idx_e[2])
            term = (dt*dz) / (r_s*dx*dy)

        eps = self.grid.eps_r[i_s, j_s, k_s, self.direction.value] * EPS_0
        sigma_e = self.grid.sigma_e[i_s, j_s, k_s, self.direction.value]

        self.grid.c_ee[i_s, j_s, k_s, self.direction.value] = \
                (2*eps - dt*sigma_e - term) / (2*eps + dt*sigma_e + term)
        self.grid.c_eh[i_s, j_s, k_s, self.direction.value] = \
                (2*dt) / (2*eps + dt*sigma_e + term)


class Capacitor(LumpedElement):
    """Model of a capacitor element.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the capacitor.
    y_min : float
        Minimum y coordinate of the bounding box containing the capacitor.
    z_min : float
        Minimum z coordinate of the bounding box containing the capacitor.
    x_max : float
        Maximum x coordinate of the bounding box containing the capacitor.
    y_max : float
        Maximum y coordinate of the bounding box containing the capacitor.
    z_max : float
        Maximum z coordinate of the bounding box containing the capacitor.
    capacitance : float
        Internal capacitance of the capacitor.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        capacitance: float = 1e-9,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.capacitance = capacitance

    def __repr__(self):
        """Dev. string representation."""
        return (f"{self.__class__.__name__}"
                f"[name={self.name}, capacitance={self.capacitance}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")

    def attach_to_grid(self):
        """Attach object to grid."""
        self.idx_s = (
            np.argmin(np.abs(self.grid.x - self.x_min)),
            np.argmin(np.abs(self.grid.y - self.y_min)),
            np.argmin(np.abs(self.grid.z - self.z_min)),
        )
        self.idx_e = (
            np.argmin(np.abs(self.grid.x - self.x_max)),
            np.argmin(np.abs(self.grid.y - self.y_max)),
            np.argmin(np.abs(self.grid.z - self.z_max)),
        )

        dx, dy, dz = self.grid.spacing
        dt = self.grid.dt
        c_s = self.capacitance

        if self.direction == Direction.X:
            i_s = slice(self.idx_s[0], self.idx_e[0])
            j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
            term = (2*c_s*dx) / (dy*dz)
        elif self.direction == Direction.Y:
            i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            j_s = slice(self.idx_s[1], self.idx_e[1])
            k_s = slice(self.idx_s[2], self.idx_e[2] + 1)
            term = (2*c_s*dy) / (dx*dz)
        else:
            i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            k_s = slice(self.idx_s[2], self.idx_e[2])
            term = (2*c_s*dz) / (dx*dy)

        eps = self.grid.eps_r[i_s, j_s, k_s, self.direction.value] * EPS_0
        sigma_e = self.grid.sigma_e[i_s, j_s, k_s, self.direction.value]

        self.grid.c_ee[i_s, j_s, k_s, self.direction.value] = \
                (2*eps - dt*sigma_e + term) / (2*eps + dt*sigma_e + term)
        self.grid.c_eh[i_s, j_s, k_s, self.direction.value] = \
                (2*dt) / (2*eps + dt*sigma_e + term)


class Inductor(LumpedElement):
    """Model of a inductor element.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the inductor.
    y_min : float
        Minimum y coordinate of the bounding box containing the inductor.
    z_min : float
        Minimum z coordinate of the bounding box containing the inductor.
    x_max : float
        Maximum x coordinate of the bounding box containing the inductor.
    y_max : float
        Maximum y coordinate of the bounding box containing the inductor.
    z_max : float
        Maximum z coordinate of the bounding box containing the inductor.
    resistance : float
        Internal resistance of the inductor.
    capacitance : float
        Internal inductance of the inductor.
    """

    def __init__(self,
                 x_min: float,
                 y_min: float,
                 z_min: float,
                 x_max: float,
                 y_max: float,
                 z_max: float,
                 resistance: float = 50,
                 inductance: float = 1e-6):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
        self.inductance = inductance


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
