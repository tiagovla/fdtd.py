from __future__ import annotations

from typing import Optional

from .bases import FDTDElementBase
from .utils import BoundingBox


class LumpedElement(FDTDElementBase):
    """A lumped element to be placed in the grid."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        name: Optional[str] = None,
    ):
        """Initialize the lumped element."""
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.name = name if name else self._create_new_name()
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)

    def attach_to_grid(self):
        """Attach object to grid."""
        pass


class Resistor(LumpedElement):
    """Implement a resistor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        super().__init__(x_min, y_min, z_min, x_max, y_max, z_max)
        self.resistance = resistance

    def __repr__(self):
        """Dev. string representation."""
        return (f"{self.__class__.__name__}"
                f"[name={self.name}, resistance={self.resistance}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")


class Capacitor(LumpedElement):
    """Implement a capacitor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


class Inductor(LumpedElement):
    """Implement a inductor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


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
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
