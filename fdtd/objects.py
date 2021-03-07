"""This module implements the objects."""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Union

import numpy as np

from .bases import FDTDElementBase
from .constants import PI
from .materials import Material
from .utils import BoundingBox

logger = logging.getLogger(__name__)


class Object(FDTDElementBase):
    """An object to be placed in the grid."""

    def __init__(self,
                 material: Union[str, Material],
                 name: Optional[str] = None):
        """Initialize the object."""
        super().__init__()

        if isinstance(material, str):
            self.material = Material.from_name(material)
        else:
            self.material = material

        self.name = name if name else self._create_new_name()

    def attach_to_grid(self):
        """Attach object to grid."""

    def plot_3d(self, ax, alpha=0.5):
        """Plot a brick and attach to an axis."""


class Brick(Object):
    """Implement a brick object."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        material: Union[str, Material],
        name: Optional[str] = None,
    ):
        """Initialize the object."""
        super().__init__(material, name)

        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)

    def __repr__(self):
        """Dev. string representation."""
        return (f"Brick[name={self.name}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")

    def attach_to_grid(self):
        """Attach the material of an object to the grid."""
        grid = self.grid
        bb = self.bounding_box
        slice_x = (grid._x_c >= bb.x_min) & (grid._x_c <= bb.x_max)
        slice_y = (grid._y_c >= bb.y_min) & (grid._y_c <= bb.y_max)
        slice_z = (grid._z_c >= bb.z_min) & (grid._z_c <= bb.z_max)
        I, J, K = np.ix_(slice_x, slice_y, slice_z)
        grid.cell_material[I, J, K, :] = [
            self.material.eps_r,
            self.material.mu_r,
            self.material.sigma_e,
            self.material.sigma_m,
        ]

    def attach_to_grid_zero_thinkness(self):
        """Attach the coeficients directly to the property grid."""
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

        sigma_e = self.grid.sigma_e
        sigma_e_mat = self.material.sigma_e

        if s[0] == e[0]:
            sigma_e[s[0], s[1]:e[1], s[2]:e[2] + 1, 1] = sigma_e_mat
            sigma_e[s[0], s[1]:e[1] + 1, s[2]:e[2], 2] = sigma_e_mat
        elif s[1] == e[1]:
            sigma_e[s[0]:e[0], s[1], s[2]:e[2] + 1, 0] = sigma_e_mat
            sigma_e[s[0]:e[0] + 1, s[1], s[2]:e[2], 2] = sigma_e_mat
        elif s[2] == e[2]:
            sigma_e[s[0]:e[0], s[1]:e[1] + 1, s[2], 0] = sigma_e_mat
            sigma_e[s[0]:e[0] + 1, s[1]:e[1], s[2], 1] = sigma_e_mat

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color=self.material.color)

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class Sphere(Object):
    """Implement a brick object."""

    def __init__(
        self,
        x_center: float,
        y_center: float,
        z_center: float,
        radius: float,
        material: Union[str, Material],
        name: Optional[str] = None,
    ):
        """Initialize the object."""
        super().__init__(material, name)
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.radius = radius

        self.bounding_box = BoundingBox(
            x_center - radius,
            x_center + radius,
            y_center - radius,
            y_center + radius,
            z_center - radius,
            z_center + radius,
        )

    def __repr__(self):
        """Dev. string representation."""
        return (f"Sphere[name={self.name}, "
                f"x_c={self.x_center}, y_c={self.y_center}, "
                f"z_c={self.z_center}, r={self.radius}]")

    def attach_to_grid(self):
        """Attach the material of an object to the grid."""
        grid = self.grid
        bb = self.bounding_box
        slice_x = (grid._x_c > bb.x_min) & (grid._x_c < bb.x_max)
        slice_y = (grid._y_c > bb.y_min) & (grid._y_c < bb.y_max)
        slice_z = (grid._z_c > bb.z_min) & (grid._z_c < bb.z_max)
        I, J, K = np.ix_(slice_x, slice_y, slice_z)
        l_m_g = grid.cell_material[I, J, K, :]
        l_x_c, l_y_c, l_z_c = np.meshgrid(grid._x_c[slice_x],
                                          grid._y_c[slice_y],
                                          grid._z_c[slice_z])
        mask = (l_x_c - self.x_center)**2 + (l_y_c - self.y_center)**2 + (
            l_z_c - self.z_center)**2 <= self.radius**2
        l_m_g[mask, :] = [
            self.material.eps_r,
            self.material.mu_r,
            self.material.sigma_e,
            self.material.sigma_m,
        ]
        grid.cell_material[I, J, K, :] = l_m_g

    def plot_3d(self, ax, alpha=0.5):
        """Plot a brick and attach to an axis."""
        u = np.linspace(0, 2 * PI, 100)
        v = np.linspace(0, PI, 100)
        x = self.x_center + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.y_center + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.z_center + self.radius * np.outer(np.ones(np.size(u)),
                                                   np.cos(v))
        ax.plot_surface(x, y, z, alpha=alpha, color=self.material.color)
