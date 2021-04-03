"""This module implements boundaries."""
from typing import Tuple, Union

import numpy as np

from .bases import FDTDElementBase
from .utils import Direction


class Boundary(FDTDElementBase):
    """Base class for all boundaries."""

    def update_E(self):
        """Update E field."""

    def update_H(self):
        """Update H field."""

    def attach_to_grid(self):
        """Attach object to grid."""


class PeriodicBoundary(Boundary):
    """Model of a periodic boundary condition.

    Parameters
    ----------
    x_direction : bool
        Flag to enable the boundary condition along the x axis.
    y_direction : bool
        Flag to enable the boundary condition along the y axis.
    z_direction : bool
        Flag to enable the boundary condition along the z axis.
    """

    def __init__(
        self,
        x_direction: bool = False,
        y_direction: bool = False,
        z_direction: bool = False,
    ):
        """Initialize periodic boundary object."""
        super().__init__()
        self.x_dir = x_direction
        self.y_dir = y_direction
        self.z_dir = z_direction

    def update_E(self):
        """Update E field."""
        if self.x_dir:
            self.grid.E[0, :, :, :] = self.grid.E[-1, :, :, :]
        if self.y_dir:
            self.grid.E[:, 0, :, :] = self.grid.E[:, -1, :, :]
        if self.z_dir:
            self.grid.E[:, :, 0, :] = self.grid.E[:, :, -1, :]

    def update_H(self):
        """Update H field."""
        if self.x_dir:
            self.grid.H[-1, :, :, :] = self.grid.H[0, :, :, :]
        if self.y_dir:
            self.grid.H[:, -1, :, :] = self.grid.H[:, 0, :, :]
        if self.z_dir:
            self.grid.H[:, :, -1, :] = self.grid.H[:, :, 0, :]


class PeriodicBlochBoundary(Boundary):
    """Model of a periodic Bloch boundary condition.

    Parameters
    ----------
    b_vec : Tuple[float,...]
    Bloch wave vector.
    x_direction : bool
        Flag to enable the boundary condition along the x axis.
    y_direction : bool
        Flag to enable the boundary condition along the y axis.
    z_direction : bool
        Flag to enable the boundary condition along the z axis.
    """

    def __init__(
        self,
        b_vec: Tuple[float, ...] = (0, 0, 0),
        x_direction: bool = False,
        y_direction: bool = False,
        z_direction: bool = False,
    ):
        """Initialize periodic Bloch boundary condition."""
        super().__init__()

        self.b_vec = b_vec
        self.x_dir = x_direction
        self.y_dir = y_direction
        self.z_dir = z_direction

        self.phi_x: np.ndarray
        self.phi_y: np.ndarray
        self.phi_z: np.ndarray

    def attach_to_grid(self):
        """Attach object to grid."""
        b_vec = np.array(self.b_vec)
        self.phi_x = np.exp(-1j * b_vec[0] *
                            (self.grid.x_max - self.grid.x_min))
        self.phi_y = np.exp(-1j * b_vec[1] *
                            (self.grid.y_max - self.grid.y_min))
        self.phi_z = np.exp(-1j * b_vec[2] *
                            (self.grid.z_max - self.grid.z_min))

    def update_E(self):
        """Update E field."""
        if self.x_dir:
            self.grid.E[0, :, :, :] = \
                    self.grid.E[-1, :, :, :] * self.phi_x.conj()
        if self.y_dir:
            self.grid.E[:, 0, :, :] = \
                    self.grid.E[:, -1, :, :] * self.phi_y.conj()
        if self.z_dir:
            self.grid.E[:, :, 0, :] = \
                    self.grid.E[:, :, -1, :] * self.phi_z.conj()

    def update_H(self):
        """Update H field."""
        if self.x_dir:
            self.grid.H[-1, :, :, :] = self.grid.H[0, :, :, :] * self.phi_x
        if self.y_dir:
            self.grid.H[:, -1, :, :] = self.grid.H[:, 0, :, :] * self.phi_y
        if self.z_dir:
            self.grid.H[:, :, -1, :] = self.grid.H[:, :, 0, :] * self.phi_z
