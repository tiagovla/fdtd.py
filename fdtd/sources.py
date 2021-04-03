"""This module implement sources."""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Tuple

import numpy as np

from .bases import FDTDElementBase
from .constants import FREESPACE_PERMITTIVITY as EPS_0
from .constants import PI
from .constants import SPEED_LIGHT as C
from .utils import BoundingBox, Direction

logger = logging.getLogger(__name__)


class Source(FDTDElementBase):
    """Base class for all sources.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the source.
    y_min : float
        Minimum y coordinate of the bounding box containing the source.
    z_min : float
        Minimum z coordinate of the bounding box containing the source.
    x_max : float
        Maximum x coordinate of the bounding box containing the source.
    y_max : float
        Maximum y coordinate of the bounding box containing the source.
    z_max : float
        Maximum z coordinate of the bounding box containing the source.
    resistance : float
        Internal resistance of the source.
    waveform_type: str
        Waveform type.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
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
        waveform_type: str = "unit_step",
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
    ):
        """Initialize the object."""
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
        self.waveform_type = waveform_type
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)
        self.direction = direction
        self.name = name if name else self._create_new_name()
        self.idx_s: Optional[Tuple] = None
        self.idx_e: Optional[Tuple] = None

        self.c_v: Optional[np.ndarray] = None
        self.i_s: Optional[slice] = None
        self.j_s: Optional[slice] = None
        self.k_s: Optional[slice] = None

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

        logger.debug(f"Source attached to {self.idx_s} to {self.idx_e}")

    def update_E(self):
        """Update E fields."""

    def update_H(self):
        """Update H fields."""

    def __repr__(self):
        """Dev. string representation."""
        return (f"{self.__class__.__name__}"
                f"[name={self.name}, waveform_type={self.waveform_type}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")

    def plot_3d(self, ax, alpha=0.5):
        """Plot a source and attach to an axis."""


class ImpressedMagneticCurrentSource(Source):
    """Model of an impressed magnetic current source.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the source.
    y_min : float
        Minimum y coordinate of the bounding box containing the source.
    z_min : float
        Minimum z coordinate of the bounding box containing the source.
    x_max : float
        Maximum x coordinate of the bounding box containing the source.
    y_max : float
        Maximum y coordinate of the bounding box containing the source.
    z_max : float
        Maximum z coordinate of the bounding box containing the source.
    resistance : float
        Internal resistance of the source.
    waveform_type: str
        Waveform type.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_H(self):
        """Update field."""
        nc = 20
        tau = (nc * np.max([self.grid.dx, self.grid.dy])) / (2*C)
        t_0 = 4.5 * tau
        source_value = np.exp(-(((self.grid.current_time - t_0) / tau)**2))
        self.grid.H[self.i_s, self.j_s, self.k_s, self.direction.value] += (
            -self.grid.c_he[self.i_s, self.j_s, self.k_s, self.direction.value]
            * source_value)

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

        logger.debug(
            f"Source {self.name} attached to {self.idx_s} to {self.idx_e}")

        self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
        self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
        self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#00FF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class ImpressedElectricCurrentSource(Source):
    """Model of an impressed electric current source.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the source.
    y_min : float
        Minimum y coordinate of the bounding box containing the source.
    z_min : float
        Minimum z coordinate of the bounding box containing the source.
    x_max : float
        Maximum x coordinate of the bounding box containing the source.
    y_max : float
        Maximum y coordinate of the bounding box containing the source.
    z_max : float
        Maximum z coordinate of the bounding box containing the source.
    resistance : float
        Internal resistance of the source.
    waveform_type: str
        Waveform type.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_E(self):
        """Update field."""
        nc = 20
        tau = (nc * np.max([self.grid.dx, self.grid.dy])) / (2*C)
        t_0 = 4.5 * tau
        source_value = np.exp(-(((self.grid.current_time - t_0) / tau)**2))
        self.grid.E[self.i_s, self.j_s, self.k_s, self.direction.value] += (
            -self.grid.c_eh[self.i_s, self.j_s, self.k_s, self.direction.value]
            * source_value)

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

        logger.debug(
            f"Source {self.name} attached to {self.idx_s} to {self.idx_e}")

        self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
        self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
        self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#00FF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class EFieldSource(Source):
    """Model of an electric field source.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the source.
    y_min : float
        Minimum y coordinate of the bounding box containing the source.
    z_min : float
        Minimum z coordinate of the bounding box containing the source.
    x_max : float
        Maximum x coordinate of the bounding box containing the source.
    y_max : float
        Maximum y coordinate of the bounding box containing the source.
    z_max : float
        Maximum z coordinate of the bounding box containing the source.
    resistance : float
        Internal resistance of the source.
    waveform_type: str
        Waveform type.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_E(self):
        """Update field."""
        nc = 20
        tau = (nc * np.max([self.grid.dx, self.grid.dy])) / (2*C)
        t_0 = 4.5 * tau
        E_s = np.exp(-(((self.grid.current_time - t_0) / tau)**2))
        self.grid.E[self.i_s, self.j_s, self.k_s, self.direction.value] += E_s

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

        logger.debug(
            f"Source {self.name} attached to {self.idx_s} to {self.idx_e}")

        self.i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
        self.j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
        self.k_s = slice(self.idx_s[2], self.idx_e[2] + 1)

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#00FF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class VoltageSource(Source):
    """Model of a voltage source.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the source.
    y_min : float
        Minimum y coordinate of the bounding box containing the source.
    z_min : float
        Minimum z coordinate of the bounding box containing the source.
    x_max : float
        Maximum x coordinate of the bounding box containing the source.
    y_max : float
        Maximum y coordinate of the bounding box containing the source.
    z_max : float
        Maximum z coordinate of the bounding box containing the source.
    resistance : float
        Internal resistance of the source.
    waveform_type: str
        Waveform type.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_E(self):
        """Update field."""
        if self.grid.current_time_step > 50:
            Vs = np.sin(2 * PI * 1e9 *
                        (self.grid.current_time - 50 * self.grid.dt))
            # Vs = 1
        else:
            Vs = 0
        # Vs = 0 if self.grid.current_time_step < 50 else 1
        Vs *= self._v_f

        if self.c_v is not None:
            if self.direction == Direction.X:
                self.grid.E[self.i_s, self.j_s, self.k_s, 0] += self.c_v * Vs
            elif self.direction == Direction.Y:
                self.grid.E[self.i_s, self.j_s, self.k_s, 1] += self.c_v * Vs
            elif self.direction == Direction.Z:
                self.grid.E[self.i_s, self.j_s, self.k_s, 2] += self.c_v * Vs

    def attach_to_grid(self):
        """Attach object to grid."""
        self.idx_s = s = (
            np.argmin(np.abs(self.grid.x - self.x_min)),
            np.argmin(np.abs(self.grid.y - self.y_min)),
            np.argmin(np.abs(self.grid.z - self.z_min)),
        )
        self.idx_e = e = (
            np.argmin(np.abs(self.grid.x - self.x_max)),
            np.argmin(np.abs(self.grid.y - self.y_max)),
            np.argmin(np.abs(self.grid.z - self.z_max)),
        )

        dx, dy, dz = self.grid.spacing
        dt = self.grid.dt
        r_s = self.resistance
        term = (dt*dz) / (r_s*dx*dy)

        if self.direction == Direction.X:
            self.i_s = i_s = slice(self.idx_s[0], self.idx_e[0])
            self.j_s = j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.k_s = k_s = slice(self.idx_s[2], self.idx_e[2] + 1)

            eps = self.grid.eps_r[i_s, j_s, k_s, 0] * EPS_0
            sigma_e = self.grid.sigma_e[i_s, j_s, k_s, 0]

            self.grid.c_ee[i_s, j_s, k_s, 0] = \
                    (2*eps - dt*sigma_e - term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[i_s, j_s, k_s, 0] = \
                    (2*dt) / (2*eps + dt*sigma_e + term)
            self.c_v = (2*dt) / (2*eps + dt*sigma_e + term) / (r_s*dy*dz)

        elif self.direction == Direction.Y:
            self.i_s = i_s = slice(self.idx_s[0], self.idx_e[0])
            self.j_s = j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.k_s = k_s = slice(self.idx_s[2], self.idx_e[2] + 1)

            eps = self.grid.eps_r[i_s, j_s, k_s, 1] * EPS_0
            sigma_e = self.grid.sigma_e[i_s, j_s, k_s, 1]

            self.grid.c_ee[i_s, j_s, k_s, 1] = \
                    (2*eps - dt*sigma_e - term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[i_s, j_s, k_s, 1] = \
                    (2*dt) / (2*eps + dt*sigma_e + term)
            self.c_v = (2*dt) / (2*eps + dt*sigma_e + term) / (r_s*dx*dz)

        else:
            self.i_s = i_s = slice(self.idx_s[0], self.idx_e[0] + 1)
            self.j_s = j_s = slice(self.idx_s[1], self.idx_e[1] + 1)
            self.k_s = k_s = slice(self.idx_s[2], self.idx_e[2])

            self._v_f = 1 / (e[2] - s[2])
            self._r_f = (e[0] - s[0] + 1) * (e[1] - s[1] + 1) * self._v_f

            eps = self.grid.eps_r[i_s, j_s, k_s, 2] * EPS_0
            sigma_e = self.grid.sigma_e[i_s, j_s, k_s, 2]
            r_s *= self._r_f
            term = (dt*dz) / (r_s*dx*dy)

            self.grid.c_ee[i_s, j_s, k_s, 2] = \
                    (2*eps - dt*sigma_e - term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[i_s, j_s, k_s, 2] = \
                    (2*dt) / (2*eps + dt*sigma_e + term)
            self.c_v = -(2 * dt) / (2*eps + dt*sigma_e + term) / (r_s*dx*dy)

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#00FF00")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class CurrentSource(Source):
    """Implement a current source."""
