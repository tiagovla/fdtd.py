from __future__ import annotations

from abc import ABC, abstractproperty
from typing import TYPE_CHECKING, Optional

from .utils import BoundingBox

if TYPE_CHECKING:
    from .grid import Grid

from .bases import FDTDElementBase


class Source(FDTDElementBase):
    """An source to be placed in the grid."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
        waveform_type: str = "unit_step",
        name: Optional[str] = None,
    ):
        """Initialize the object."""
        super().__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
        self.waveform_type = waveform_type
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min,
                                        z_max)
        self.name = name if name else self._create_new_name()

    def attach_to_grid(self):
        """Attach object to grid."""
        pass

    def __repr__(self):
        """Dev. string representation."""
        return (f"{self.__class__.__name__}"
                f"[name={self.name}, waveform_type={self.waveform_type}, "
                f"x_min={self.x_min}, y_min={self.y_min}, z_min={self.z_min}, "
                f"x_max={self.x_max}, y_max={self.y_max}, z_max={self.z_max}]")


class VoltageSource(Source):
    """Implement a voltage source."""

    pass


class CurrentSource(Source):
    """Implement a current source."""

    pass
