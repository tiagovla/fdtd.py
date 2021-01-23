"""This module implements the utilities."""
from dataclasses import dataclass

from numpy import ndarray


@dataclass
class BoundingBox:
    """Implement a bounding box that contains an object."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
