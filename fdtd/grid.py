"""This module implements the grid."""
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .constants import FREESPACE_PERMEABILITY, FREESPACE_PERMITTIVITY, PI, SPEED_LIGHT
from .objects import Brick, Object, Sphere
from .source import Source


class Grid:
    """Implement the simulation grid."""

    def __init__(
        self,
        shape: Tuple[int, int, int],
        grid_spacing: Union[float, Tuple[float, float, float]],
        permittivity: float = 1,
        permeability: float = 1,
        courant_number: float = 1,
    ):
        """Initialize the grid."""
        if isinstance(grid_spacing, float):
            self.grid_spacing = (grid_spacing, grid_spacing, grid_spacing)
        else:
            self.grid_spacing = grid_spacing

        self.Nx, self.Ny, self.Nz = shape
        self.courant_number = courant_number
        self.time_step = self.courant_number * min(
            self.grid_spacing) / SPEED_LIGHT

        self.E = np.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.H = np.zeros((self.Nx, self.Ny, self.Nz, 3))

        self.permittivity = np.ones(
            (self.Nx, self.Ny, self.Nz, 3)) * permittivity
        self.permeability = np.ones(
            (self.Nx, self.Ny, self.Nz, 3)) * permeability

        # Objects and Sources:
        self.sources: List[Source] = []
        self.objects: List[Object] = []
        self.current_time_step: int = 0

        # Spacial limits:
        self._x = self.grid_spacing[0] * np.arange(0, self.Nx + 1)
        self._y = self.grid_spacing[1] * np.arange(0, self.Ny + 1)
        self._z = self.grid_spacing[2] * np.arange(0, self.Nz + 1)

    # TODO: Add center so the domain.
    @property
    def x_min(self) -> float:
        """Return x_min of the grid domain."""
        return 0

    @property
    def y_min(self) -> float:
        """Return y_min of the grid domain."""
        return 0

    @property
    def z_min(self) -> float:
        """Return z_min of the grid domain."""
        return 0

    @property
    def x_max(self) -> float:
        """Return x_max of the grid domain."""
        return self.Nx * self.grid_spacing[0]

    @property
    def y_max(self) -> float:
        """Return y_max of the grid domain."""
        return self.Ny * self.grid_spacing[1]

    @property
    def z_max(self) -> float:
        """Return z_max of the grid domain."""
        return self.Nz * self.grid_spacing[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the grid."""
        return (self.Nx, self.Ny, self.Nz)

    @property
    def time_passed(self) -> float:
        """Return time passed since the start of the simulation."""
        return self.current_time_step * self.time_step

    def run(self, total_time: float):
        """Run simulation."""
        for _ in np.arange(0, np.floor(total_time * self.time_step), 1):
            self.step()

    def step(self):
        """Run a single step of the simulation."""
        self.update_E()
        self.update_H()
        self.current_time_step += 1

    def update_E(self):
        """Update E Field."""
        pass

    def update_H(self):
        """Update H Field."""
        pass

    def add_source(self, source):
        """Add source to the grid."""
        self.sources.append(source)

    def add_object(self, object):
        """Add object to the grid."""
        self.objects.append(object)

    def plot_3d(self) -> None:
        """Plot grid."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for obj in self.objects:
            color = obj.material.color
            if isinstance(obj, Sphere):
                u = np.linspace(0, 2 * PI, 100)
                v = np.linspace(0, PI, 100)
                x = obj.x_center + obj.radius * np.outer(np.cos(u), np.sin(v))
                y = obj.y_center + obj.radius * np.outer(np.sin(u), np.sin(v))
                z = obj.z_center + obj.radius * np.outer(
                    np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.5, color=color)
            if isinstance(obj, Brick):
                X, Y, Z = np.meshgrid(
                    [obj.x_min, obj.x_max],
                    [obj.y_min, obj.y_max],
                    [obj.z_min, obj.z_max],
                )
                ax.plot_surface(X[:, :, 0],
                                Y[:, :, 0],
                                Z[:, :, 0],
                                alpha=0.5,
                                color=color)
                ax.plot_surface(X[:, :, -1],
                                Y[:, :, -1],
                                Z[:, :, -1],
                                alpha=0.5,
                                color=color)
                ax.plot_surface(X[:, 0, :],
                                Y[:, 0, :],
                                Z[:, 0, :],
                                alpha=0.5,
                                color=color)
                ax.plot_surface(X[:, -1, :],
                                Y[:, -1, :],
                                Z[:, -1, :],
                                alpha=0.5,
                                color=color)
                ax.plot_surface(X[0, :, :],
                                Y[0, :, :],
                                Z[0, :, :],
                                alpha=0.5,
                                color=color)
                ax.plot_surface(X[-1, :, :],
                                Y[-1, :, :],
                                Z[-1, :, :],
                                alpha=0.5,
                                color=color)
        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.grid_spacing[0]])
        ax.set_ylim([0, self.Ny * self.grid_spacing[1]])
        ax.set_zlim([0, self.Nz * self.grid_spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        plt.show()
