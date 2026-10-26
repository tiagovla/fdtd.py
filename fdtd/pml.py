"""Convolutional perfectly matched layers on the exterior grid faces."""

from numbers import Integral, Real

import numpy as np

from .boundaries import Boundary
from .constants import FREESPACE_PERMITTIVITY as EPS_0


class CPMLBoundary(Boundary):
    """Absorb outgoing waves in slabs inside selected exterior faces.

    ``thickness`` is the number of cells per slab. ``faces`` contains names
    such as ``("x-", "x+")``; by default both faces of every active axis are
    selected. The slabs end at electric conducting walls on the outer nodes.
    ``alpha_min`` and ``alpha_max`` have units of electric conductivity (S/m).
    """

    def __init__(
        self,
        thickness=10,
        *,
        faces=None,
        order=3,
        sigma_factor=1.1,
        kappa_max=5,
        alpha_min=0,
        alpha_max=0.05,
    ):
        super().__init__()
        if (
            isinstance(thickness, bool)
            or not isinstance(thickness, Integral)
            or thickness < 1
        ):
            raise ValueError("CPML thickness must be a positive integer.")
        for name, value in (
            ("order", order),
            ("sigma_factor", sigma_factor),
            ("kappa_max", kappa_max),
            ("alpha_min", alpha_min),
            ("alpha_max", alpha_max),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"CPML {name} must be finite and real.")
        if order <= 0 or sigma_factor <= 0 or kappa_max < 1:
            raise ValueError(
                "CPML requires positive order/sigma_factor and kappa_max >= 1."
            )
        if not 0 <= alpha_min <= alpha_max:
            raise ValueError("CPML requires 0 <= alpha_min <= alpha_max.")
        if faces is not None:
            try:
                faces = (faces,) if isinstance(faces, str) else tuple(faces)
            except TypeError:
                raise ValueError(
                    "CPML faces must be a face name or an iterable of names."
                ) from None
            valid = ("x-", "x+", "y-", "y+", "z-", "z+")
            if not faces or any(
                not isinstance(face, str) or face not in valid for face in faces
            ):
                raise ValueError(
                    "CPML faces must be selected from x-, x+, y-, y+, z-, z+."
                )
            if len(set(faces)) != len(faces):
                raise ValueError("CPML faces must not be repeated.")
        self.thickness = int(thickness)
        self.faces = faces
        self.order = order
        self.sigma_factor = sigma_factor
        self.kappa_max = kappa_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._layers = []

    def _selected_faces(self):
        if self.faces is None:
            return tuple(
                "xyz"[axis] + side
                for axis, size in enumerate(self.grid.shape)
                if size > 1
                for side in ("-", "+")
            )
        return self.faces

    def attach_to_grid(self):
        faces = self._selected_faces()
        occupied = self.grid._cpml_faces.copy()
        for face in faces:
            axis = "xyz".index(face[0])
            if self.grid.shape[axis] == 1:
                raise ValueError("CPML faces must belong to active grid axes.")
            if self.grid._periodic[axis]:
                raise ValueError("CPML and periodic boundaries cannot share an axis.")
            if face in occupied:
                raise ValueError("Only one CPML boundary may configure each face.")
            occupied[face] = self.thickness
            total = sum(occupied.get(face[0] + side, 0) for side in ("-", "+"))
            if total >= self.grid.shape[axis] - 1:
                raise ValueError(
                    "CPML slabs must leave at least one interior cell per axis."
                )
        self._layers = [_CPMLLayer(self, face) for face in faces]
        self.grid._cpml_faces = occupied

    def reset(self):
        self._layers = []

    def estimate_memory(self):
        total = 0
        for face in self._selected_faces():
            axis = "xyz".index(face[0])
            cells = self.thickness * np.prod(
                [size for i, size in enumerate(self.grid.shape) if i != axis]
            )
            total += 2 * cells * (self.grid.E.itemsize + self.grid.H.itemsize)
            total += 6 * self.thickness * np.dtype(float).itemsize
        return int(total)

    def correct_electric_curl(self, curl):
        for layer in self._layers:
            layer.correct_curl(curl, self.grid.H, forward=False)

    def correct_magnetic_curl(self, curl):
        for layer in self._layers:
            layer.correct_curl(curl, self.grid.E, forward=True)

    def update_E(self):
        for layer in self._layers:
            wall = [slice(None)] * 3
            wall[layer.axis] = 0 if layer.lower else -1
            for component in range(3):
                if component != layer.axis:
                    self.grid.E[tuple(wall) + (component,)] = 0


class _CPMLLayer:
    _terms = (
        ((2, 1, -1), (1, 2, 1)),
        ((2, 0, 1), (0, 2, -1)),
        ((1, 0, -1), (0, 1, 1)),
    )

    def __init__(self, boundary, face):
        grid = boundary.grid
        self.axis = "xyz".index(face[0])
        self.lower = face[1] == "-"
        self.spacing = grid.spacing[self.axis]
        n = boundary.thickness
        start = 0 if self.lower else grid.shape[self.axis] - n - 1

        def region(first):
            index = [slice(None)] * 3
            index[self.axis] = slice(first, first + n)
            return tuple(index)

        self.here_h = region(start)
        self.next_e = region(start + 1)
        self.here_e = region(1 if self.lower else start)
        self.previous_h = region(0 if self.lower else start - 1)
        shape = list(grid.shape)
        shape[self.axis] = n
        self.psi_e = np.zeros(tuple(shape) + (2,), dtype=grid.E.dtype)
        self.psi_h = np.zeros(tuple(shape) + (2,), dtype=grid.H.dtype)

        depth = np.arange(n, 0, -1) if self.lower else np.arange(1, n + 1)
        rho_e = (depth - 0.75) / n
        rho_h = (depth - 0.25) / n
        self.coefficients_e = self._coefficients(boundary, rho_e)
        self.coefficients_h = self._coefficients(boundary, rho_h)

    def _coefficients(self, boundary, rho):
        sigma_max = (
            boundary.sigma_factor * (boundary.order + 1) / (150 * np.pi * self.spacing)
        )
        grading = rho**boundary.order
        sigma = sigma_max * grading
        kappa = 1 + (boundary.kappa_max - 1) * grading
        alpha = boundary.alpha_min + (boundary.alpha_max - boundary.alpha_min) * (
            1 - rho
        )
        decay = np.expm1(-boundary.grid.dt / EPS_0 * (sigma / kappa + alpha))
        denominator = kappa * (sigma + kappa * alpha)
        a = np.divide(
            decay * sigma, denominator, out=np.zeros_like(sigma), where=denominator != 0
        )
        shape = [1, 1, 1]
        shape[self.axis] = len(rho)
        return tuple(value.reshape(shape) for value in (1 / kappa - 1, a, decay + 1))

    def correct_curl(self, curl, field, *, forward):
        if forward:
            here, other = self.here_h, self.next_e
            profiles, memory = self.coefficients_h, self.psi_h
        else:
            here, other = self.here_e, self.previous_h
            profiles, memory = self.coefficients_e, self.psi_e
        stretch, a, b = profiles
        for index, (component, target, sign) in enumerate(self._terms[self.axis]):
            difference = field[other + (component,)] - field[here + (component,)]
            difference *= (1 if forward else -1) / self.spacing
            psi = memory[..., index]
            psi *= b
            psi += a * difference
            curl[here + (target,)] += sign * (stretch * difference + psi)
