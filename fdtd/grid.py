"""This module implements the grid."""
from __future__ import annotations

import logging
from math import floor
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .constants import FREESPACE_PERMEABILITY as MU_0
from .constants import FREESPACE_PERMITTIVITY as EPS_0
from .constants import SPEED_LIGHT
from .lumped_elements import LumpedElement
from .objects import Brick, Object
from .sources import Source

if TYPE_CHECKING:
    from .objects import Object

logger = logging.getLogger(__name__)


class Grid:
    """Implement the simulation grid."""

    def __init__(
        self,
        shape: Tuple[int, int, int],
        grid_spacing: Union[float, Tuple[float, float, float]],
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

        self.dt = 0.1  # TODO: Define it

        # Field sizes:
        #  H_x (Nx+1, Ny  , Nz  ) H_y (Nx  , Ny+1, Nz  ) H_z (Nx  , Ny  , Nz+1)
        #  E_x (Nx  , Ny+1, Nz+1) E_y (Nx+1, Ny  , Nz+1) E_z (Nx+1, Ny+1, Nz  )

        self.E = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.H = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))

        self.cell_material: np.ndarray = np.concatenate(
            (
                np.ones((self.Nx, self.Ny, self.Nz, 2)),
                1e-20 * np.ones((self.Nx, self.Ny, self.Nz, 2)),
            ),
            axis=3,
        )

        # Property sizes:
        #   eps_r_x   (Nx  , Ny+1, Nz+1)
        #   eps_r_y   (Nx+1, Ny  , Nz+1)
        #   eps_r_z   (Nx+1, Ny+1, Nz  )
        #   mu_r_x    (Nx+1, Ny  , Nz  )
        #   mu_r_y    (Nx  , Ny+1, Nz  )
        #   mu_r_z    (Nx  , Ny  , Nz+1)
        #   sigma_e_x (Nx  , Ny+1, Nz+1)
        #   sigma_e_y (Nx+1, Ny  , Nz+1)
        #   sigma_e_z (Nx+1, Ny+1, Nz  )
        #   sigma_m_x (Nx+1, Ny  , Nz  )
        #   sigma_m_y (Nx  , Ny+1, Nz  )
        #   sigma_m_z (Nx  , Ny  , Nz+1)

        # Properties:
        self.eps_r = np.ones((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.mu_r = np.ones((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.sigma_e = 1e-20 + np.zeros(
            (self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.sigma_m = 1e-20 + np.zeros(
            (self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))

        # Updating coefficients:
        # E(t+dt) = c_ee*E(t)+c_eh*curl_H(t+dt)
        # H(t+dt) = c_hh*H(t)+c_he*curl_E(t+dt)
        self.c_ee = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.c_eh = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.c_hh = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))
        self.c_he = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz + 1, 3))

        # Objects and Sources:
        self.sources: List[Source] = []
        self.objects: List[Object] = []
        self.lumped_elements: List[LumpedElement] = []
        self.current_time_step: int = 0
        self.current_time: float = 0

        # Spacial center limits:
        self._x_c: np.ndarray = self.grid_spacing[0] * (0.5 +
                                                        np.arange(0, self.Nx))
        self._y_c: np.ndarray = self.grid_spacing[1] * (0.5 +
                                                        np.arange(0, self.Ny))
        self._z_c: np.ndarray = self.grid_spacing[2] * (0.5 +
                                                        np.arange(0, self.Nz))
        self._x: np.ndarray = self.grid_spacing[0] * np.arange(0, self.Nx + 1)
        self._y: np.ndarray = self.grid_spacing[1] * np.arange(0, self.Ny + 1)
        self._z: np.ndarray = self.grid_spacing[2] * np.arange(0, self.Nz + 1)

    def _calculate_material_components(self):
        cel_m = self.cell_material
        # yapf: disable
        self.eps_r[0:-1, 1:-1, 1:-1, 0] = \
            0.25 * (
                cel_m[:, 1:, 1:, 0] + cel_m[:, :-1, 1:, 0] +
                cel_m[:, 1:, :-1, 0] + cel_m[:, :-1, :-1, 0]
            )
        self.eps_r[1:-1, 0:-1, 1:-1, 1] = \
            0.25 * (
                cel_m[1:, :, 1:, 0] + cel_m[:-1, :, 1:, 0] +
                cel_m[1:, :, :-1, 0] + cel_m[:-1, :, :-1, 0]
            )
        self.eps_r[1:-1, 1:-1, 0:-1, 2] = \
            0.25 * (
                cel_m[1:, 1:, :, 0] + cel_m[:-1, 1:, :, 0] +
                cel_m[1:, :-1, :, 0] + cel_m[:-1, :-1, :, 0]
            )

        self.sigma_e[0:-1, 1:-1, 1:-1, 0] = \
            0.25 * (
                cel_m[:, 1:, 1:, 3] + cel_m[:, :-1, 1:, 3] +
                cel_m[:, 1:, :-1, 3] + cel_m[:, :-1, :-1, 3]
            )
        self.sigma_e[1:-1, 0:-1, 1:-1, 1] = \
            0.25 * (
                cel_m[1:, :, 1:, 3] + cel_m[:-1, :, 1:, 3] +
                cel_m[1:, :, :-1, 3] + cel_m[:-1, :, :-1, 3]
            )
        self.sigma_e[1:-1, 1:-1, 0:-1, 2] = \
            0.25 * (
                cel_m[1:, 1:, :, 3] + cel_m[:-1, 1:, :, 3] +
                cel_m[1:, :-1, :, 3] + cel_m[:-1, :-1, :, 3]
            )

        self.mu_r[1:-1, 0:-1, 0:-1, 0] = \
            2 * (cel_m[1:, :, :, 1]*cel_m[:-1, :, :, 1]) / \
            (cel_m[1:, :, :, 1] + cel_m[:-1, :, :, 1])

        self.mu_r[0:-1, 1:-1, 0:-1, 1] = \
            2 * (cel_m[:, 1:, :, 1]*cel_m[:, :-1, :, 1]) / \
            (cel_m[:, 1:, :, 1] + cel_m[:, :-1, :, 1])

        self.mu_r[0:-1, 0:-1, 1:-1, 2] = \
            2 * (cel_m[:, :, 1:, 1]*cel_m[:, :, :-1, 1]) / \
            (cel_m[:, :, 1:, 1] + cel_m[:, :, :-1, 1])

        self.sigma_m[1:-1, 0:-1, 0:-1, 0] = \
            2 * (cel_m[1:, :, :, 3]*cel_m[:-1, :, :, 3]) / \
            (cel_m[1:, :, :, 3] + cel_m[:-1, :, :, 3])

        self.sigma_m[0:-1, 1:-1, 0:-1, 1] = \
            2 * (cel_m[:, 1:, :, 3]*cel_m[:, :-1, :, 3]) / \
            (cel_m[:, 1:, :, 3] + cel_m[:, :-1, :, 3])

        self.sigma_m[0:-1, 0:-1, 1:-1, 2] = \
            2 * (cel_m[:, :, 1:, 3]*cel_m[:, :, :-1, 3]) / \
            (cel_m[:, :, 1:, 3] + cel_m[:, :, :-1, 3])
        # yapf: enable

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

    def run(self, n_steps: int = 1000, total_time: Optional[float] = None):
        """Run simulation."""
        self._prepare()
        self._initialize_updating_coefficients()

        if total_time:
            n_steps = floor(total_time / self.time_step)

        logger.info(f"Running simulation with {n_steps} steps")

        for _ in range(0, n_steps):
            self.step()

    def step(self):
        """Run a single step of the simulation."""
        self.update_H()
        self.update_E()
        self.current_time_step += 1

    def update_sources(self):
        """Update sources."""
        pass

    def update_elements(self):
        """Update elements."""
        pass

    def update_E(self):
        """Update E Field."""
        self.current_time += self.dt / 2
        dx, dy, dz = self.grid_spacing
        self.E *= self.c_ee
        self.E += self.c_eh * self.curl_H(self.H, dx, dy, dz)

    def update_H(self):
        """Update H Field."""
        self.current_time += self.dt / 2
        dx, dy, dz = self.grid_spacing
        self.H *= self.c_hh
        self.H += self.c_he * self.curl_E(self.E, dx, dy, dz)

    def add(self, elm: Union[Object, Source, LumpedElement]):
        """Add element to grid."""
        elm._register_grid(self)
        if isinstance(elm, Object):
            self.objects.append(elm)
        elif isinstance(elm, Source):
            self.sources.append(elm)
        elif isinstance(elm, LumpedElement):
            self.lumped_elements.append(elm)

    def _prepare(self):
        """Prepare grid, add objects, sources, ..."""
        logger.info("Preparing objects, sources, lumped elements, ...")
        for obj in self.objects:
            obj.attach_to_grid()

        for source in self.sources:
            source.attach_to_grid()

        self._calculate_material_components()
        for obj in self.objects:
            if isinstance(obj, Brick):
                obj.attach_to_grid_zero_thinkness()
        self._initialize_updating_coefficients()

    def plot_planes(self) -> None:
        """Plot material."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # X, Y = np.meshgrid(np.arange(0, self.Nx + 1),
        #                    np.arange(0, self.Ny + 1))
        # ax.plot_surface(X, Y, self.eps_r[:, :, 5, 0])
        X, Y, Z = np.meshgrid(
            np.arange(0, self.Nx + 1),
            np.arange(0, self.Ny + 1),
            np.arange(0, self.Nz + 1),
        )
        ax.scatter(X, Y, Z, c=self.eps_r[:, :, :, 2])
        plt.show()

    def _initialize_updating_coefficients(self):
        logger.info("Initializing updating coefficients...")
        dx, dy, dz = self.grid_spacing
        dt = self.dt
        eps = self.eps_r * EPS_0
        mu = self.mu_r * MU_0

        f_e = (dt * self.sigma_e) / (2*eps)
        f_m = (dt * self.sigma_m) / (2*mu)

        self.c_ee = (1-f_e) / (1+f_e)
        self.c_eh = dt / (eps * (1+f_e))
        self.c_hh = (1-f_m) / (1+f_m)
        self.c_he = -dt / (mu * (1+f_e))

    def plot_disc2(self) -> None:
        """Plot material."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = np.meshgrid(self._x, self._y, self._z)
        mask = self.sigma_e[:, :, :, 1] > 0.101
        ax.scatter(X[mask], Y[mask], Z[mask])
        # print(self.sigma_e[:, :, :, 0])

        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.grid_spacing[0]])
        ax.set_ylim([0, self.Ny * self.grid_spacing[1]])
        ax.set_zlim([0, self.Nz * self.grid_spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        # plt.show()

    def plot_disc(self) -> None:
        """Plot material."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = np.meshgrid(self._x_c, self._y_c, self._z_c)
        mask = self.cell_material[:, :, :, 2] > 0.001
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

    @staticmethod
    def curl_H(H: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """curl_H.

        Parameters
        ----------
        H : np.ndarray
            H field.
        dx : float
            Discrete spacing along the x axis.
        dy : float
            Discrete spacing along the y axis.
        dz : float
            Discrete spacing along the z axis.

        Returns
        -------
        curl : np.ndarray
            Return the curl of H.

        """
        curl = np.zeros(H.shape)
        curl[:, 1:, :, 0] += (H[:, 1:, :, 2] - H[:, :-1, :, 2]) / dy
        curl[:, :, 1:, 0] -= (H[:, :, 1:, 1] - H[:, :, :-1, 1]) / dz

        curl[:, :, 1:, 1] += (H[:, :, 1:, 0] - H[:, :, :-1, 0]) / dz
        curl[1:, :, :, 1] -= (H[1:, :, :, 2] - H[:-1, :, :, 2]) / dx

        curl[1:, :, :, 2] += (H[1:, :, :, 1] - H[:-1, :, :, 1]) / dx
        curl[:, 1:, :, 2] -= (H[:, 1:, :, 0] - H[:, :-1, :, 0]) / dy

        return curl

    @staticmethod
    def curl_E(E: np.ndarray, dx: float, dy: float, dz: float):
        """curl_E.

        Parameters
        ----------
        E : np.ndarray
            E field.
        dx : float
            Discrete spacing along the x axis.
        dy : float
            Discrete spacing along the y axis.
        dz : float
            Discrete spacing along the z axis.

        Returns
        -------
        curl : np.ndarray
            Return the curl of E.

        """
        curl = np.zeros(E.shape)
        curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2]) / dy
        curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1]) / dz

        curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0]) / dz
        curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2]) / dx

        curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1]) / dx
        curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0]) / dy

        return curl
