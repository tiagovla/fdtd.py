"""This module implements the objects."""
from abc import ABC, abstractproperty
from functools import partial
from typing import Union

import numpy as np

from .constants import PI
from .materials import Material
from .utils import BoundingBox


class Object(ABC):
    """An object to be placed in the grid."""

    def __init__(self):
        """Initialize the object."""
        self.grid = None

    @abstractproperty
    def name(self) -> str:
        """Return the name of the object."""

    def __repr__(self):
        """Dev. string representation."""
        return f"{Object}({self.name})"

    def register_grid(self, grid):
        """Register grid."""
        self.grid = grid

    def attach_to_grid(self):
        """Attach object to grid."""
        pass


class Brick(Object):
    """Implement a brick object."""

    name = "Brick"

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        material: Union[str, Material],
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        if isinstance(material, str):
            self.material = Material.from_name(material)
        else:
            self.material = material
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)

    def attach_to_grid(self):
        """Attach the material of an object to the grid."""
        slice_x = (self.grid._x_c > self.x_min) & (self.grid._x_c < self.x_max)
        slice_y = (self.grid._y_c > self.y_min) & (self.grid._y_c < self.y_max)
        slice_z = (self.grid._z_c > self.z_min) & (self.grid._z_c < self.z_max)
        self.grid.cell_material[slice_x[:, None, None] *
                                slice_y[None, :, None] *
                                slice_z[None, None, :], :] = [
                                    self.material.eps_r,
                                    self.material.mu_r,
                                    self.material.sigma_e,
                                    self.material.sigma_m,
                                ]

    def plot_3d(self, ax, alpha=0.5):
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

    name = "Sphere"

    def __init__(
        self,
        x_center: float,
        y_center: float,
        z_center: float,
        radius: float,
        material: Union[str, Material],
    ):
        """Initialize the object."""
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.radius = radius
        if isinstance(material, str):
            self.material = Material.from_name(material)
        else:
            self.material = material
        self.bounding_box = BoundingBox(
            x_center - radius,
            x_center + radius,
            y_center - radius,
            y_center + radius,
            z_center - radius,
            z_center + radius,
        )

    def attach_to_grid(self):
        """Attach the material of an object to the grid."""
        x_min = self.x_center - self.radius
        x_max = self.x_center + self.radius
        y_min = self.x_center - self.radius
        y_max = self.x_center + self.radius
        z_min = self.x_center - self.radius
        z_max = self.x_center + self.radius
        slice_x = (self.grid._x_c > x_min) & (self.grid._x_c < x_max)
        slice_y = (self.grid._y_c > y_min) & (self.grid._y_c < y_max)
        slice_z = (self.grid._z_c > z_min) & (self.grid._z_c < z_max)
        I, J, K = np.ix_(slice_x, slice_y, slice_z)
        local_grid = self.grid.cell_material[I, J, K, :]

        l_x_c, l_y_c, l_z_c = np.meshgrid(self.grid._x_c[slice_x],
                                          self.grid._y_c[slice_y],
                                          self.grid._z_c[slice_z])
        mask = (l_x_c - self.x_center)**2 + (l_y_c - self.y_center)**2 + (
            l_z_c - self.z_center)**2 <= self.radius**2

        local_grid[mask, :] = [1, 2, 3, 4]
        self.grid.cell_material[I, J, K, :] = local_grid

    def plot_3d(self, ax, alpha=0.5):
        """Plot a brick and attach to an axis."""
        u = np.linspace(0, 2 * PI, 100)
        v = np.linspace(0, PI, 100)
        x = self.x_center + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.y_center + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.z_center + self.radius * np.outer(np.ones(np.size(u)),
                                                   np.cos(v))
        ax.plot_surface(x, y, z, alpha=alpha, color=self.material.color)
