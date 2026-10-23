"""This module implements the grid."""

from __future__ import annotations

import logging
from math import floor
from numbers import Integral, Real
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .boundaries import Boundary
from .constants import FREESPACE_PERMEABILITY as MU_0
from .constants import FREESPACE_PERMITTIVITY as EPS_0
from .constants import SPEED_LIGHT
from .detectors import Detector
from .lumped_elements import LumpedElement
from .objects import Brick, Object
from .plotting import pyplot
from .sources import Source
from .utils import curl_E, curl_H, validate_node_bounds

if TYPE_CHECKING:
    from .objects import Object

logger = logging.getLogger(__name__)


class Grid:
    """Implement the simulation grid."""

    def __init__(
        self,
        shape: Tuple[int, int, int],
        spacing: Union[float, Tuple[float, float, float]],
        courant_factor: float = 0.9,
    ):
        """Initialize the grid."""
        if (
            len(shape) != 3
            or any(
                isinstance(n, bool) or not isinstance(n, Integral) or n < 1
                for n in shape
            )
            or max(shape) == 1
        ):
            raise ValueError(
                "shape must contain three positive integers and an active axis."
            )
        spacing = (spacing,) * 3 if isinstance(spacing, Real) else tuple(spacing)
        if len(spacing) != 3 or any(
            not isinstance(d, Real)
            or isinstance(d, bool)
            or not np.isfinite(d)
            or d <= 0
            for d in spacing
        ):
            raise ValueError("spacing must contain three finite positive values.")
        self.spacing = tuple(float(d) for d in spacing)

        self.dx, self.dy, self.dz = self.spacing
        self.Nx, self.Ny, self.Nz = shape
        self.dim = sum([1 for s in self.shape if s > 1])

        if (
            isinstance(courant_factor, Real)
            and np.isfinite(courant_factor)
            and 0 < courant_factor < 1
        ):
            self.courant_factor = courant_factor
        else:
            raise ValueError(f"courant_factor {courant_factor} must be between 0 and 1")

        self.dt = self.courant_factor / (
            SPEED_LIGHT
            * np.sqrt(
                np.float_power(
                    [
                        s1 if s2 > 1 else np.inf
                        for s1, s2 in zip(self.spacing, self.shape)
                    ],
                    -2,
                ).sum()
            )
        )

        self.E = np.zeros((self.Nx, self.Ny, self.Nz, 3), dtype=complex)
        self.H = np.zeros((self.Nx, self.Ny, self.Nz, 3), dtype=complex)

        self.cell_material: np.ndarray = np.concatenate(
            (
                np.ones((self.Nx, self.Ny, self.Nz, 2)),
                1e-20 * np.ones((self.Nx, self.Ny, self.Nz, 2)),
            ),
            axis=3,
        )
        self._background_material = None

        # Properties:
        self.eps_r = np.ones((self.Nx, self.Ny, self.Nz, 3))
        self.mu_r = np.ones((self.Nx, self.Ny, self.Nz, 3))
        self.sigma_e = 1e-20 + np.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.sigma_m = 1e-20 + np.zeros((self.Nx, self.Ny, self.Nz, 3))

        # Updating coefficients:
        # E(t+dt) = c_ee*E(t)+c_eh*curl_H(t+dt)
        # H(t+dt) = c_hh*H(t)+c_he*curl_E(t+dt)
        self.c_ee = np.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.c_eh = np.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.c_hh = np.zeros((self.Nx, self.Ny, self.Nz, 3))
        self.c_he = np.zeros((self.Nx, self.Ny, self.Nz, 3))

        # Objects and Sources:
        self.sources: List[Source] = []
        self.objects: List[Object] = []
        self.detectors: List[Detector] = []
        self.lumped_elements: List[LumpedElement] = []
        self.boundaries: List[Boundary] = []
        self.current_time_step: int = 0
        self.current_time: float = 0
        self.n_steps: int = 0
        self._prepared = False
        self._periodic = [False, False, False]
        self._phases = np.ones(3, dtype=complex)

        # usefull auxiliary arrays:
        self.x_c: np.ndarray = self.dx * (0.5 + np.arange(0, self.Nx))
        self.y_c: np.ndarray = self.dy * (0.5 + np.arange(0, self.Ny))
        self.z_c: np.ndarray = self.dz * (0.5 + np.arange(0, self.Nz))
        self.x: np.ndarray = self.dx * np.arange(0, self.Nx)
        self.y: np.ndarray = self.dy * np.arange(0, self.Ny)
        self.z: np.ndarray = self.dz * np.arange(0, self.Nz)

    def reset(self):
        """Reset the grid's inner state."""
        self.current_time_step = 0
        self.current_time = 0
        self.n_steps = 0
        self._prepared = False
        self.E.fill(0)
        self.H.fill(0)
        if self._background_material is not None:
            self.cell_material[:] = self._background_material
            self._background_material = None
        for element in (
            *self.sources,
            *self.lumped_elements,
            *self.detectors,
            *self.boundaries,
        ):
            element.reset()

        self.c_ee[:] = 0
        self.c_eh[:] = 0
        self.c_hh[:] = 0
        self.c_he[:] = 0

        self.eps_r[:] = 1
        self.mu_r[:] = 1
        self.sigma_e[:] = 1e-20
        self.sigma_m[:] = 1e-20

    def _calculate_material_components_simple(self):
        """Calculate material components."""
        self.eps_r[:, :, :, :] = self.cell_material[:, :, :, 0, None]
        self.mu_r[:, :, :, :] = self.cell_material[:, :, :, 1, None]
        self.sigma_e[:, :, :, :] = self.cell_material[:, :, :, 2, None]
        self.sigma_m[:, :, :, :] = self.cell_material[:, :, :, 3, None]

    def _calculate_material_components(self):
        """Average cells at electric edges and magnetic faces."""
        periodic = getattr(self, "_periodic", (False,) * 3)

        def previous(values, axis):
            shifted = np.roll(values, 1, axis=axis)
            if not periodic[axis]:
                index = [slice(None)] * 3
                index[axis] = 0
                shifted[tuple(index)] = values[tuple(index)]
            return shifted

        for component in range(3):
            for target, column in ((self.eps_r, 0), (self.sigma_e, 2)):
                values = self.cell_material[..., column].copy()
                for axis in range(3):
                    if axis != component and self.shape[axis] > 1:
                        values = (values + previous(values, axis)) / 2
                target[..., component] = values
            for target, column in ((self.mu_r, 1), (self.sigma_m, 3)):
                values = self.cell_material[..., column]
                adjacent = previous(values, component)
                total = values + adjacent
                target[..., component] = np.divide(
                    2 * values * adjacent,
                    total,
                    out=np.zeros_like(values),
                    where=total != 0,
                )

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
        return self.Nx * self.dx

    @property
    def y_max(self) -> float:
        """Return y_max of the grid domain."""
        return self.Ny * self.dy

    @property
    def z_max(self) -> float:
        """Return z_max of the grid domain."""
        return self.Nz * self.dz

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the grid."""
        return (self.Nx, self.Ny, self.Nz)

    @property
    def time_passed(self) -> float:
        """Return time passed since the start of the simulation."""
        return self.current_time_step * self.dt

    def _prepare(self):
        """Prepare grid, add objects, sources, ..."""
        logger.info("Preparing objects, sources, lumped elements, ...")
        if self._background_material is None:
            self._background_material = self.cell_material.copy()
        else:
            self.cell_material[:] = self._background_material
        for obj in self.objects:
            obj.attach_to_grid()

        if (
            self.cell_material.shape != self.shape + (4,)
            or not np.all(np.isfinite(self.cell_material))
            or np.any(self.cell_material[..., :2] <= 0)
            or np.any(self.cell_material[..., 2:] < 0)
        ):
            raise ValueError(
                "Material grid requires positive eps/mu and nonnegative conductivities."
            )
        for element in (*self.sources, *self.lumped_elements, *self.detectors):
            validate_node_bounds(self, element)

        self._periodic = [False, False, False]
        self._phases = np.ones(3, dtype=complex)
        for bc in self.boundaries:
            bc.attach_to_grid()

        self._calculate_material_components()

        for obj in self.objects:
            if isinstance(obj, Brick):
                obj.attach_to_grid_zero_thinkness()

        self._initialize_updating_coefficients()

        for source in self.sources:
            source.attach_to_grid()

        for elm in self.lumped_elements:
            elm.attach_to_grid()

        for det in self.detectors:
            det.attach_to_grid()
        self._prepared = True

    def run(
        self,
        n_steps: int = 1000,
        total_time: Optional[float] = None,
        *,
        restart: bool = False,
        progress: bool = True,
    ):
        """Advance by n_steps (or total_time), retaining previous samples.

        Use restart=True, or call reset(), to start again from zero fields.
        """
        if total_time is None:
            if (
                isinstance(n_steps, bool)
                or not isinstance(n_steps, Integral)
                or n_steps <= 0
            ):
                raise ValueError("n_steps must be a positive integer.")
            steps = int(n_steps)
        else:
            if (
                not isinstance(total_time, Real)
                or not np.isfinite(total_time)
                or total_time <= 0
            ):
                raise ValueError("total_time must be finite and positive.")
            steps = floor(total_time / self.dt + 1e-12)
            if steps < 1:
                raise ValueError("total_time must cover at least one time step.")
        if restart:
            self.reset()
        self.n_steps = self.current_time_step + steps

        logger.info("Preparing sources, objects, detectors...")
        if not self._prepared:
            self._prepare()
        else:
            for detector in self.detectors:
                detector.attach_to_grid()

        logger.info("Running simulation with %s steps...", steps)

        for _ in tqdm(range(steps), disable=not progress):
            self.step()

        logger.info("Processing results...")
        self._results()

    def _results(self):
        for det in self.detectors:
            det.plot()

    def estimate_memory(self, n_steps):
        arrays = (
            self.E,
            self.H,
            self.cell_material,
            self.eps_r,
            self.mu_r,
            self.sigma_e,
            self.sigma_m,
            self.c_ee,
            self.c_eh,
            self.c_hh,
            self.c_he,
        )
        return (
            sum(array.nbytes for array in arrays)
            + self.cell_material.nbytes
            + sum(detector.estimate_memory(n_steps) for detector in self.detectors)
        )

    def step(self):
        """Run a single step of the simulation."""
        if not self._prepared:
            self._prepare()
        if self.detectors and self.current_time_step >= self.n_steps:
            raise RuntimeError("Detector storage exhausted; advance with Grid.run().")
        self.update_H()
        for bc in self.boundaries:
            bc.update_H()

        self.update_E()
        for bc in self.boundaries:
            bc.update_E()

        for det in self.detectors:
            det.update()
        self.current_time_step += 1

    def update_E(self):
        """Update E Field."""
        self.current_time += self.dt / 2
        dx, dy, dz = self.spacing
        self.E *= self.c_ee
        self.E += self.c_eh * curl_H(self.H, dx, dy, dz, self._periodic, self._phases)

        for src in self.sources:
            src.update_E()
        for element in self.lumped_elements:
            element.update_E()

    def update_H(self):
        """Update H Field."""
        self.current_time += self.dt / 2
        dx, dy, dz = self.spacing
        self.H *= self.c_hh
        self.H += self.c_he * curl_E(self.E, dx, dy, dz, self._periodic, self._phases)

        for src in self.sources:
            src.update_H()

    def add(self, elm: Union[Object, Source, LumpedElement, Detector, Boundary]):
        """Add element to grid."""
        if not isinstance(elm, (Object, Source, LumpedElement, Detector, Boundary)):
            raise TypeError("Unsupported grid element.")
        if self.current_time_step:
            raise RuntimeError("Reset the grid before changing its components.")
        if isinstance(elm, (Source, LumpedElement, Detector)):
            validate_node_bounds(self, elm)
        elm._register_grid(self)
        if isinstance(elm, Object):
            self.objects.append(elm)
        elif isinstance(elm, Source):
            self.sources.append(elm)
        elif isinstance(elm, LumpedElement):
            self.lumped_elements.append(elm)
        elif isinstance(elm, Detector):
            self.detectors.append(elm)
        elif isinstance(elm, Boundary):
            self.boundaries.append(elm)
        self._prepared = False

    def remove(self, element):
        if self.current_time_step:
            raise RuntimeError("Reset the grid before changing its components.")
        for collection in (
            self.objects,
            self.sources,
            self.lumped_elements,
            self.detectors,
            self.boundaries,
        ):
            if element in collection:
                collection.remove(element)
                element.grid = None
                element.reset()
                self._prepared = False
                return
        raise ValueError("Element is not registered with this grid.")

    def plot_planes(self) -> None:
        """Plot material."""
        plt = pyplot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # X, Y = np.meshgrid(np.arange(0, self.Nx + 1),
        #                    np.arange(0, self.Ny + 1))
        # ax.plot_surface(X, Y, self.eps_r[:, :, 5, 0])
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        ax.scatter(X, Y, Z, c=self.eps_r[:, :, :, 2])
        plt.show()

    def _initialize_updating_coefficients(self):
        """Initialize updating coefficients."""

        logger.info("Initializing updating coefficients...")
        dt = self.dt
        eps = self.eps_r * EPS_0
        mu = self.mu_r * MU_0

        f_e = (dt * self.sigma_e) / (2 * eps)
        f_m = (dt * self.sigma_m) / (2 * mu)

        self.c_ee = (1 - f_e) / (1 + f_e)
        self.c_eh = dt / (eps * (1 + f_e))
        self.c_hh = (1 - f_m) / (1 + f_m)
        self.c_he = -dt / (mu * (1 + f_m))

        logger.debug(f"c_ee {self.c_ee[0, 0, 0, 0]} {self.c_ee.shape}")
        logger.debug(f"c_eh {self.c_eh[0, 0, 0, 0]} {self.c_eh.shape}")
        logger.debug(f"c_hh {self.c_hh[0, 0, 0, 0]} {self.c_hh.shape}")
        logger.debug(f"c_he {self.c_he[0, 0, 0, 0]} {self.c_he.shape}")

    def plot_disc2(self):
        """Plot material."""
        plt = pyplot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        mask = self.eps_r[:, :, :, 1] > 1
        x, y, z = np.nonzero(mask)
        ax.scatter(x * self.spacing[0], y * self.spacing[1], z * self.spacing[2])

        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.spacing[0]])
        ax.set_ylim([0, self.Ny * self.spacing[1]])
        ax.set_zlim([0, self.Nz * self.spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        plt.show()

    def plot_disc3(self):
        """Plot material."""
        plt = pyplot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        mask = self.eps_r[:, :, :, 2] > 1
        ax.scatter(X[mask], Y[mask], Z[mask])

        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.spacing[0]])
        ax.set_ylim([0, self.Ny * self.spacing[1]])
        ax.set_zlim([0, self.Nz * self.spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        plt.show()

    def plot_disc(self) -> None:
        """Plot material."""
        plt = pyplot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        X, Y, Z = np.meshgrid(self.x_c, self.y_c, self.z_c, indexing="ij")
        mask = self.cell_material[:, :, :, 0] > 1
        ax.scatter(X[mask], Y[mask], Z[mask])

        ax.grid(True)
        ax.set_xlim([0, self.Nx * self.spacing[0]])
        ax.set_ylim([0, self.Ny * self.spacing[1]])
        ax.set_zlim([0, self.Nz * self.spacing[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        plt.show()

    def plot_3d(self, z_view: bool = False):
        """Plot grid."""
        plt = pyplot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for obj in self.objects:
            obj.plot_3d(ax)

        for src in self.sources:
            src.plot_3d(ax)

        for elm in self.lumped_elements:
            elm.plot_3d(ax)

        for det in self.detectors:
            det.plot_3d(ax)

        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        mask = self.sigma_e[:, :, :, 0] > 0.1
        ax.scatter(X[mask] + self.dx / 2, Y[mask], Z[mask], c="blue")

        mask = self.sigma_e[:, :, :, 1] > 0.1
        ax.scatter(X[mask], Y[mask] + self.dy / 2, Z[mask], c="red")

        ax.grid(True)
        ax.set_xticks(self.x)
        ax.set_yticks(self.y)
        ax.set_zticks(self.z)
        ax.set_xlim([0, self.Nx * self.dx])
        ax.set_ylim([0, self.Ny * self.dy])
        ax.set_zlim([0, self.Nz * self.dz])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=25, azim=-135)
        if z_view:
            ax.view_init(elev=90, azim=-90)
        plt.show()
