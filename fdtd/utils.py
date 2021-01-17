"""This module implements the utilities."""
from dataclasses import dataclass

from numpy import ndarray


class LocalMaterialGrid:
    """Implement local grid."""

    def __init__(self, cell_material: ndarray, x_c: ndarray, y_c: ndarray,
                 z_c: ndarray):
        """Initialize local grid."""
        self.cell_material = cell_material
        self.x_c = x_c
        self.y_c = y_c
        self.z_c = z_c


@dataclass
class BoundingBox:
    """Implement a bounding box that contains an object."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
