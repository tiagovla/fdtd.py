"""This module implements the grid."""
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .constants import FREESPACE_PERMEABILITY, FREESPACE_PERMITTIVITY, PI, SPEED_LIGHT
from .objects import Brick, Object, Sphere
from .source import Source
from .utils import BoundingBox, LocalMaterialGrid


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

        self.E: np.ndarray = np.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.H: np.ndarray = np.zeros((self.Nx, self.Ny, self.Nz, 3))

        self.permittivity = np.ones(
            (self.Nx, self.Ny, self.Nz, 3)) * permittivity
        self.permeability = np.ones(
            (self.Nx, self.Ny, self.Nz, 3)) * permeability

        self.cell_material: np.ndarray = np.concatenate(
            (
                np.ones((self.Nx, self.Ny, self.Nz, 2)),
                np.zeros((self.Nx, self.Ny, self.Nz, 2)),
            ),
            axis=3,
        )
        # Properties
        self.eps_r_x = np.ones((self.Nx, self.Ny + 1, self.Nz + 1))
        self.eps_r_y = np.ones((self.Nx + 1, self.Ny, self.Nz + 1))
        self.eps_r_z = np.ones((self.Nx + 1, self.Ny + 1, self.Nz))
        self.mu_r_x = np.ones((self.Nx + 1, self.Ny, self.Nz))
        self.mu_r_y = np.ones((self.Nx, self.Ny + 1, self.Nz))
        self.mu_r_z = np.ones((self.Nx, self.Ny + 1, self.Nz + 1))
        self.sigma_e_x = np.ones((self.Nx, self.Ny + 1, self.Nz + 1))
        self.sigma_e_y = np.ones((self.Nx + 1, self.Ny, self.Nz + 1))
        self.sigma_e_z = np.ones((self.Nx + 1, self.Ny + 1, self.Nz))
        self.sigma_m_x = np.ones((self.Nx + 1, self.Ny, self.Nz))
        self.sigma_m_y = np.ones((self.Nx, self.Ny + 1, self.Nz))
        self.sigma_m_z = np.ones((self.Nx, self.Ny + 1, self.Nz + 1))
        # Objects and Sources:
        self.sources: List[Source] = []
        self.objects: List[Object] = []
        self.current_time_step: int = 0

        # Spacial center limits:
        self._x_c: np.ndarray = self.grid_spacing[0] * (0.5 +
                                                        np.arange(0, self.Nx))
        self._y_c: np.ndarray = self.grid_spacing[1] * (0.5 +
                                                        np.arange(0, self.Ny))
        self._z_c: np.ndarray = self.grid_spacing[2] * (0.5 +
                                                        np.arange(0, self.Nz))

    # TODO: Add center so the domain.
    def average_parameters(self) -> None:
        """Average the surrounding parameters."""
        pass

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

    def prepare(self):
        """Prepare grid, add objects, sources, ..."""
        for obj in self.objects:
            bb = obj.bounding_box
            slice_x = (self._x_c > bb.x_min) & (self._x_c < bb.x_max)
            slice_y = (self._y_c > bb.y_min) & (self._y_c < bb.y_max)
            slice_z = (self._z_c > bb.z_min) & (self._z_c < bb.z_max)
            I, J, K = np.ix_(slice_x, slice_y, slice_z)
            lmg = LocalMaterialGrid(
                self.cell_material[I, J, K, :].copy(),
                self._x_c[slice_x],
                self._y_c[slice_y],
                self._z_c[slice_z],
            )
            obj.attach_to_grid(lmg)
            self.cell_material[I, J, K, :] = lmg.cell_material

    def plot_disc(self) -> None:
        """Plot material."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = np.meshgrid(self._x_c, self._y_c, self._z_c)
        mask = self.cell_material[:, :, :, 0] != 1
        ax.scatter(X[mask], Y[mask], Z[mask])

        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.grid_spacing[0]])
        ax.set_ylim([0, self.Ny * self.grid_spacing[1]])
        ax.set_zlim([0, self.Nz * self.grid_spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        plt.show()

    def plot_3d(self) -> None:
        """Plot grid."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for obj in self.objects:
            obj.plot_3d(ax)
        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.grid_spacing[0]])
        ax.set_ylim([0, self.Ny * self.grid_spacing[1]])
        ax.set_zlim([0, self.Nz * self.grid_spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        plt.show()