"""This module implements the objects."""
from abc import ABC, abstractproperty
from typing import Union

from .materials import Material


class Object(ABC):
    """An object to be placed in the grid."""

    def __init__(self):
        """Initialize the object."""
        pass

    @abstractproperty
    def name(self) -> str:
        """Return the name of the object."""

    def __repr__(self):
        """Dev. string representation."""
        return f"{Object}({self.name})"


class Brick(Object):
    """Implement a brick object."""

    name = "Brick"

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        material: Union[str, Material],
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        if isinstance(material, str):
            self.material = Material[material]
        else:
            self.material = material


class Sphere(Object):
    """Implement a brick object."""

    name = "Sphere"

    def __init__(
        self,
        x_center: float,
        y_center: float,
        z_center: float,
        radius: float,
        material: Union[str, Material],
    ):
        """Initialize the object."""
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.radius = radius
        if isinstance(material, str):
            self.material = Material[material]
        else:
            self.material = material
