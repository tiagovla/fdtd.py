"""Boundary conditions for wrapped staggered derivatives."""

from abc import abstractmethod

import numpy as np

from .bases import FDTDElementBase


class Boundary(FDTDElementBase):
    @abstractmethod
    def attach_to_grid(self):
        """Configure derivative behavior before field updates."""

    def update_E(self):
        """Wrapped derivatives already enforce the electric boundary."""

    def update_H(self):
        """Wrapped derivatives already enforce the magnetic boundary."""


class PeriodicBoundary(Boundary):
    def __init__(self, x_direction=False, y_direction=False, z_direction=False):
        super().__init__()
        self.x_dir, self.y_dir, self.z_dir = x_direction, y_direction, z_direction

    def attach_to_grid(self):
        phases = self._phase_factors()
        for axis, enabled in enumerate((self.x_dir, self.y_dir, self.z_dir)):
            if enabled:
                if self.grid._periodic[axis]:
                    raise ValueError(
                        "Only one periodic boundary may configure each axis."
                    )
                self.grid._periodic[axis] = True
                self.grid._phases[axis] = phases[axis]

    def _phase_factors(self):
        return np.ones(3, dtype=complex)


class PeriodicBlochBoundary(PeriodicBoundary):
    def __init__(
        self, b_vec=(0, 0, 0), x_direction=False, y_direction=False, z_direction=False
    ):
        super().__init__(x_direction, y_direction, z_direction)
        self.b_vec = b_vec

    def _phase_factors(self):
        vector = np.asarray(self.b_vec, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("b_vec must contain three finite wave-vector components.")
        phases = np.exp(-1j * vector * np.asarray(self.grid.shape) * self.grid.spacing)
        self.phi_x, self.phi_y, self.phi_z = phases
        return phases
