"""This module implements the utilities."""
from dataclasses import dataclass


class LocalGrid:
    """Implement local grid."""

    pass


@dataclass
class BoundingBox:
    """Implement a boundary box that contains an object."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
