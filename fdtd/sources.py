"""This module implement sources."""

from __future__ import annotations

import logging
from abc import abstractmethod
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .bases import FDTDElementBase
from .utils import BoundingBox, Direction
from .waveforms import Waveform

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
    waveform : Union[Waveform, Callable]
        Waveform type.
    resistance : float
        Internal resistance of the source.
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
        waveform: Union[Waveform, Callable],
        resistance: float = 50,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
    ):
        """Initialize the object."""
        super().__init__()
        if not isinstance(direction, Direction):
            raise ValueError("direction must be a Direction value.")
        if not callable(waveform):
            raise ValueError("waveform must be callable.")
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
        self.waveform = waveform
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)
        self.direction = direction
        self.name = name if name else self._create_new_name()
        self.idx_s: Optional[Tuple] = None
        self.idx_e: Optional[Tuple] = None

        self.c_v: Optional[np.ndarray] = None
        self.i_s: Optional[slice] = None
        self.j_s: Optional[slice] = None
        self.k_s: Optional[slice] = None

    @abstractmethod
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

    def estimate_memory(self):
        """Return additional cached source storage in bytes."""
        return 0

    def __repr__(self):
        """Dev. string representation."""
        return (
            f"{self.__class__.__name__}"
            f"[name={self.name}, waveform={self.waveform}, "
            f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]"
        )

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
    waveform : Union[Waveform, Callable]
        Waveform type.
    resistance : float
        Internal resistance of the source.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_H(self):
        """Update field."""
        source_value = self.waveform(self.grid.current_time - self.grid.dt / 2)
        self.grid.H[self.i_s, self.j_s, self.k_s, self.direction.value] += (
            self.grid.c_he[self.i_s, self.j_s, self.k_s, self.direction.value]
            * source_value
        )

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

        logger.debug(f"Source {self.name} attached to {self.idx_s} to {self.idx_e}")

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
    waveform : Union[Waveform, Callable]
        Waveform type.
    resistance : float
        Internal resistance of the source.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_E(self):
        """Update field."""
        source_value = self.waveform(self.grid.current_time - self.grid.dt / 2)
        self.grid.E[self.i_s, self.j_s, self.k_s, self.direction.value] += (
            -self.grid.c_eh[self.i_s, self.j_s, self.k_s, self.direction.value]
            * source_value
        )

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

        logger.debug(f"Source {self.name} attached to {self.idx_s} to {self.idx_e}")

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
    waveform : Union[Waveform, Callable]
        Waveform type.
    resistance : float
        Internal resistance of the source.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_E(self):
        """Update field."""
        E_s = self.waveform(self.grid.current_time)
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

        logger.debug(f"Source {self.name} attached to {self.idx_s} to {self.idx_e}")

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
    waveform : Union[Waveform, Callable]
        Waveform type.
    resistance : float
        Internal resistance of the source.
    name : Optional[str]
        Name of the source.
    direction : Direction
        Direction of the source.
    """

    def update_E(self):
        """Update field."""
        if self.c_v is None:
            return
        index = (self.i_s, self.j_s, self.k_s, self.direction.value)
        # Use the final coupling after all parallel R/C loads have attached.
        self.c_v = -self.grid.c_eh[index] / self._resistance_area
        voltage = self._v_f * self.waveform(self.grid.current_time - self.grid.dt / 2)
        self.grid.E[index] += self.c_v * voltage

    def attach_to_grid(self):
        """Attach object to grid."""
        super().attach_to_grid()
        axis = self.direction.value
        counts = np.subtract(self.idx_e, self.idx_s)
        series = counts[axis]
        if series <= 0:
            raise ValueError("Voltage source must span at least one field edge.")
        if not np.isfinite(self.resistance) or self.resistance < 0:
            raise ValueError("Source resistance must be finite and nonnegative.")
        parallel = np.prod([counts[d] + 1 for d in range(3) if d != axis])
        self._v_f = 1 / series
        self._r_f = parallel / series
        resistance = max(self.resistance, 1e-20) * self._r_f
        area = np.prod([self.grid.spacing[d] for d in range(3) if d != axis])
        self._resistance_area = resistance * area
        self.i_s, self.j_s, self.k_s = (
            slice(s, e + (d != axis))
            for d, (s, e) in enumerate(zip(self.idx_s, self.idx_e))
        )
        index = (self.i_s, self.j_s, self.k_s, axis)
        self.grid._add_electric_load(
            index, conductivity=self.grid.spacing[axis] / self._resistance_area
        )
        self.c_v = -self.grid.c_eh[index] / self._resistance_area

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
