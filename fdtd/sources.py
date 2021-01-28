"""This module implement sources."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .bases import FDTDElementBase
from .constants import FREESPACE_PERMITTIVITY as EPS_0
from .utils import BoundingBox, Direction

logger = logging.getLogger(__name__)


class Source(FDTDElementBase):
    """An source to be placed in the grid."""

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
        logger.debug("Source initial and final snapped point")
        logger.debug(self.idx_s)
        logger.debug(self.idx_e)

    def __repr__(self):
        """Dev. string representation."""
        return (f"{self.__class__.__name__}"
                f"[name={self.name}, waveform_type={self.waveform_type}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")


class VoltageSource(Source):
    """Implement a voltage source."""

    def update(self):

        pass

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
        logger.debug("Source initial and final snapped point")
        logger.debug(self.idx_s)
        logger.debug(self.idx_e)

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
            self.c_v = (2*dt) / (2*eps + dt*sigma_e + term) / (Rs*dy*dz)

        elif self.direction == Direction.Y:
            I = slice(self.idx_s[0], self.idx_e[0])
            J = slice(self.idx_s[1], self.idx_e[1] + 1)
            K = slice(self.idx_s[2], self.idx_e[2] + 1)

            eps = self.grid.eps_r[I, J, K, 1] * EPS_0
            sigma_e = self.grid.sigma_e[I, J, K, 1]

            self.grid.c_ee[I, J, K, 1] = (2*eps - dt*sigma_e -
                                          term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 0] = (2*dt) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 2] = self.grid.c_eh[I, J, K, 0]
            self.c_v = (2*dt) / (2*eps + dt*sigma_e + term) / (Rs*dx*dz)

        else:
            I = slice(self.idx_s[0], self.idx_e[0] + 1)
            J = slice(self.idx_s[1], self.idx_e[1] + 1)
            K = slice(self.idx_s[2], self.idx_e[2])

            eps = self.grid.eps_r[I, J, K, 2] * EPS_0
            sigma_e = self.grid.sigma_e[I, J, K, 2]

            self.grid.c_ee[I, J, K, 2] = (2*eps - dt*sigma_e -
                                          term) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 1] = (2*dt) / (2*eps + dt*sigma_e + term)
            self.grid.c_eh[I, J, K, 0] = self.grid.c_eh[I, J, K, 1]
            self.c_v = (2*dt) / (2*eps + dt*sigma_e + term) / (Rs*dx*dy)


class CurrentSource(Source):
    """Implement a current source."""

    pass
