"""This module implement lumped elements."""

from __future__ import annotations

import logging
from abc import abstractmethod
from functools import partial
from typing import Optional

import numpy as np

from .bases import FDTDElementBase
from .utils import BoundingBox, Direction

logger = logging.getLogger(__name__)


class LumpedElement(FDTDElementBase):
    """Base class for all lumped elements.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the lumped element.
    y_min : float
        Minimum y coordinate of the bounding box containing the lumped element.
    z_min : float
        Minimum z coordinate of the bounding box containing the lumped element.
    x_max : float
        Maximum x coordinate of the bounding box containing the lumped element.
    y_max : float
        Maximum y coordinate of the bounding box containing the lumped element.
    z_max : float
        Maximum z coordinate of the bounding box containing the lumped element.
    name : Optional[str]
        Name of the lumped element.
    direction : Direction
        Direction of the lumped element.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
    ):
        """Initialize the lumped element."""
        super().__init__()
        if not isinstance(direction, Direction):
            raise ValueError("direction must be a Direction value.")
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.name = name if name else self._create_new_name()
        self.direction = direction

        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)

    @abstractmethod
    def attach_to_grid(self):
        """Attach object to grid."""
        raise NotImplementedError

    def update_E(self):
        """Static elements are already included in the update coefficients."""

    def _set_region(self):
        coordinates = (self.grid.x, self.grid.y, self.grid.z)
        self.idx_s = tuple(
            int(np.argmin(abs(a - b)))
            for a, b in zip(coordinates, (self.x_min, self.y_min, self.z_min))
        )
        self.idx_e = tuple(
            int(np.argmin(abs(a - b)))
            for a, b in zip(coordinates, (self.x_max, self.y_max, self.z_max))
        )
        axis = self.direction.value
        self.series = self.idx_e[axis] - self.idx_s[axis]
        self.parallel = np.prod(
            [self.idx_e[d] - self.idx_s[d] + 1 for d in range(3) if d != axis]
        )
        if self.series <= 0:
            raise ValueError("Lumped element must span at least one field edge.")
        self.slices = tuple(
            slice(s, e + (d != axis))
            for d, (s, e) in enumerate(zip(self.idx_s, self.idx_e))
        )
        self.index = self.slices + (axis,)
        self.edge_length = self.grid.spacing[axis]
        self.edge_area = np.prod([self.grid.spacing[d] for d in range(3) if d != axis])

    def plot_3d(self, ax, alpha: float = 0.5):
        """Plot a brick and attach to an axis."""
        X, Y, Z = np.meshgrid(
            [self.x_min, self.x_max],
            [self.y_min, self.y_max],
            [self.z_min, self.z_max],
        )
        plot = partial(ax.plot_surface, alpha=alpha, color="#ff0000")

        plot(X[:, :, 0], Y[:, :, 0], Z[:, :, 0])
        plot(X[:, :, -1], Y[:, :, -1], Z[:, :, -1])
        plot(X[:, 0, :], Y[:, 0, :], Z[:, 0, :])
        plot(X[:, -1, :], Y[:, -1, :], Z[:, -1, :])
        plot(X[0, :, :], Y[0, :, :], Z[0, :, :])
        plot(X[-1, :, :], Y[-1, :, :], Z[-1, :, :])


class Resistor(LumpedElement):
    """Model of a resistor element.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the resistor.
    y_min : float
        Minimum y coordinate of the bounding box containing the resistor.
    z_min : float
        Minimum z coordinate of the bounding box containing the resistor.
    x_max : float
        Maximum x coordinate of the bounding box containing the resistor.
    y_max : float
        Maximum y coordinate of the bounding box containing the resistor.
    z_max : float
        Maximum z coordinate of the bounding box containing the resistor.
    resistance : float
        Internal resistance of the resistor.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
    ):
        """Initialize the object."""
        super().__init__(x_min, y_min, z_min, x_max, y_max, z_max, name, direction)
        self.resistance = resistance

    def __repr__(self):
        """Dev. string representation."""
        return (
            f"{self.__class__.__name__}"
            f"[name={self.name}, resistance={self.resistance}, "
            f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]"
        )

    def attach_to_grid(self):
        """Attach object to grid."""
        if not np.isfinite(self.resistance) or self.resistance <= 0:
            raise ValueError("Resistance must be finite and positive.")
        self._set_region()
        resistance = self.resistance * self.parallel / self.series
        self.grid._add_electric_load(
            self.index, conductivity=self.edge_length / (resistance * self.edge_area)
        )


class Capacitor(LumpedElement):
    """Model of a capacitor element.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the capacitor.
    y_min : float
        Minimum y coordinate of the bounding box containing the capacitor.
    z_min : float
        Minimum z coordinate of the bounding box containing the capacitor.
    x_max : float
        Maximum x coordinate of the bounding box containing the capacitor.
    y_max : float
        Maximum y coordinate of the bounding box containing the capacitor.
    z_max : float
        Maximum z coordinate of the bounding box containing the capacitor.
    capacitance : float
        Internal capacitance of the capacitor.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        capacitance: float = 1e-9,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
    ):
        """Initialize the object."""
        super().__init__(x_min, y_min, z_min, x_max, y_max, z_max, name, direction)
        self.capacitance = capacitance

    def __repr__(self):
        """Dev. string representation."""
        return (
            f"{self.__class__.__name__}"
            f"[name={self.name}, capacitance={self.capacitance}, "
            f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]"
        )

    def attach_to_grid(self):
        """Attach object to grid."""
        if not np.isfinite(self.capacitance) or self.capacitance <= 0:
            raise ValueError("Capacitance must be finite and positive.")
        self._set_region()
        capacitance = self.capacitance * self.series / self.parallel
        self.grid._add_electric_load(
            self.index, permittivity=capacitance * self.edge_length / self.edge_area
        )


class Inductor(LumpedElement):
    """Model of a inductor element.

    Parameters
    ----------
    x_min : float
        Minimum x coordinate of the bounding box containing the inductor.
    y_min : float
        Minimum y coordinate of the bounding box containing the inductor.
    z_min : float
        Minimum z coordinate of the bounding box containing the inductor.
    x_max : float
        Maximum x coordinate of the bounding box containing the inductor.
    y_max : float
        Maximum y coordinate of the bounding box containing the inductor.
    z_max : float
        Maximum z coordinate of the bounding box containing the inductor.
    resistance : float
        Internal resistance of the inductor.
    inductance : float
        Internal inductance of the inductor.
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 0,
        inductance: float = 1e-6,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
    ):
        """Initialize the object."""
        super().__init__(x_min, y_min, z_min, x_max, y_max, z_max, name, direction)
        self.resistance = resistance
        self.inductance = inductance
        self.current_density = None

    def reset(self):
        self.current_density = None

    def attach_to_grid(self):
        if not np.isfinite(self.inductance) or self.inductance <= 0:
            raise ValueError("Inductance must be finite and positive.")
        if not np.isfinite(self.resistance) or self.resistance < 0:
            raise ValueError("Series resistance must be finite and nonnegative.")
        self._set_region()
        scale = self.parallel / self.series
        inductance = self.inductance * scale
        resistance = self.resistance * scale
        damping = self.grid.dt * resistance / (2 * inductance)
        self._decay = (1 - damping) / (1 + damping)
        self._drive = (
            self.grid.dt
            * self.edge_length
            / (inductance * self.edge_area * (1 + damping))
        )
        self.current_density = np.zeros_like(self.grid.E[self.index])

    def update_E(self):
        self.grid.E[self.index] -= self.grid.c_eh[self.index] * self.current_density
        self.current_density = (
            self._decay * self.current_density + self._drive * self.grid.E[self.index]
        )


class Diode(LumpedElement):
    """Implement a diode element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 0,
        name: Optional[str] = None,
        direction: Direction = Direction.Z,
        *,
        saturation_current: float = 1e-14,
        temperature: float = 300,
        ideality: float = 1,
        polarity: int = 1,
        tolerance: float = 1e-12,
        max_iterations: int = 100,
    ):
        """Initialize the object."""
        super().__init__(x_min, y_min, z_min, x_max, y_max, z_max, name, direction)
        self.resistance = resistance
        self.saturation_current = saturation_current
        self.temperature = temperature
        self.ideality = ideality
        self.polarity = polarity
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.previous_voltage = 0.0

    def reset(self):
        self.previous_voltage = 0.0

    def attach_to_grid(self):
        for name in ("saturation_current", "temperature", "ideality", "tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.resistance) or self.resistance < 0:
            raise ValueError("Series resistance must be finite and nonnegative.")
        if self.polarity not in (-1, 1):
            raise ValueError("Diode polarity must be -1 or 1.")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        self._set_region()
        if self.series != 1 or self.parallel != 1:
            raise ValueError("A diode must occupy a single field edge.")
        self._thermal_voltage = (
            self.ideality * 1.380649e-23 * self.temperature / 1.602176634e-19
        )
        self._gamma = (
            float(self.grid.c_eh[self.index].item()) * self.edge_length / self.edge_area
        )
        self.previous_voltage = 0.0

    def _junction_voltage(self, midpoint):
        vt = self._thermal_voltage
        a = (self.resistance + self._gamma / 2) * self.saturation_current / vt
        b = midpoint / vt
        lower, upper = min(b, 0.0), max(b, 0.0)
        x = min(b, np.log1p(b / a)) if b > 0 else b
        for _ in range(self.max_iterations):
            exponent = np.exp(min(x, 700.0))
            with np.errstate(over="ignore", invalid="ignore"):
                residual = x + a * np.expm1(min(x, 700.0)) - b
                derivative = 1 + a * exponent
            if abs(residual) * vt <= self.tolerance * (1 + abs(midpoint)):
                return x * vt
            if residual > 0:
                upper = x
            else:
                lower = x
            candidate = x - residual / derivative if np.isfinite(residual) else np.nan
            x = candidate if lower < candidate < upper else (lower + upper) / 2
        raise RuntimeError("Diode junction solve did not converge.")

    def update_E(self):
        field = self.grid.E[self.index].item()
        if not np.isfinite(field) or field.imag != 0:
            raise ValueError("Nonlinear diodes require finite real time-domain fields.")
        # Later parallel loads can change the final electric coupling.
        self._gamma = (
            float(self.grid.c_eh[self.index].item()) * self.edge_length / self.edge_area
        )
        predicted = self.polarity * self.edge_length * field.real
        midpoint = (self.previous_voltage + predicted) / 2
        junction = self._junction_voltage(midpoint)
        current = (midpoint - junction) / (self.resistance + self._gamma / 2)
        voltage = predicted - self._gamma * current
        self.grid.E[self.index] = self.polarity * voltage / self.edge_length
        self.previous_voltage = voltage
