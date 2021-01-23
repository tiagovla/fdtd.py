"""This module implements the objects."""
import json
from typing import List

from .exceptions import MaterialExists, MaterialNotFound


class MaterialMeta(type):
    """Material metaclass."""

    @property
    def all(cls) -> List["Material"]:
        """Return list of all materials registered."""
        return list(cls.__dict__["__materials__"])

    def __prepare__(cls, name):
        """Implement prepare function."""
        return {"__materials__": set()}

    def __getitem__(self, name: str):
        """Return a material given a name."""
        for material in self.__dict__["__materials__"]:
            if name == material.name:
                return material
        raise MaterialNotFound(f"Material {name} not found.")


class Material(metaclass=MaterialMeta):
    """The material of an object."""

    def __init__(
        self,
        name: str,
        eps_r: float = 1,
        mu_r: float = 1,
        sigma_e: float = 0,
        sigma_m: float = 0,
        color: str = "#000000",
    ):
        """Initialize the object."""
        self.name = name
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma_e = sigma_e
        self.sigma_m = sigma_m
        self.color = color
        self._register(self)

    def _register(self, material) -> None:
        materials_set = self.__class__.__dict__["__materials__"]
        if material not in materials_set:
            materials_set.add(material)
        else:
            raise MaterialExists(f"Material {material.name} already exists.")

    def _unregister(self, material):
        materials_set = self.__class__.__dict__["__materials__"]
        materials_set.discard(material)

    def delete(self):
        """Delete material."""
        self._unregister(self)

    def __del__(self):
        """Implement destructor."""
        self.delete()

    @classmethod
    def load(cls, file_name):
        """Load materials from .json file."""
        with open(file_name, "r") as f:
            data = json.load(f)
        for props in data:
            cls(**props)

    @classmethod
    def from_name(cls, name: str):
        """Initialize material from a name."""
        for material in cls.__dict__["__materials__"]:
            if name == material.name:
                return material
        raise MaterialNotFound(f"Material {name} not found.")

    def __hash__(self) -> int:
        """Implement hash function."""
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """Implement equal function."""
        return self.name == other.name

    def __repr__(self) -> str:
        """Representation of the material."""
        return f"Material({self.name})"
