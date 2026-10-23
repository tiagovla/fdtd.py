"""This module implements the objects."""

from __future__ import annotations

import json
from typing import List

import numpy as np

from .exceptions import MaterialExists, MaterialNotFound


class MaterialMeta(type):
    """Material metaclass."""

    @property
    def all(cls) -> List["Material"]:
        """Return list of all materials registered."""
        return list(cls.__dict__["__materials__"].values())

    def __new__(mcls, name, bases, namespace):
        namespace["__materials__"] = {}
        return super().__new__(mcls, name, bases, namespace)

    def __getitem__(cls, name: str):
        """Return a material given a name."""
        try:
            return cls.__dict__["__materials__"][name]
        except KeyError:
            raise MaterialNotFound(f"Material {name} not found.") from None


class Material(metaclass=MaterialMeta):
    """Implement a material.

    Parameters
    ----------
    name : str
        The name of the material.
    eps_r : float
        Relative permissivity.
    mu_r : float
        Relative permeability.
    sigma_e : float
        Electric condutivity.
    sigma_m : float
        Magnetic condutivity.
    color : str
        RGB color.
    """

    def __init__(
        self,
        name: str,
        eps_r: float = 1,
        mu_r: float = 1,
        sigma_e: float = 0,
        sigma_m: float = 0,
        color: str = "#000000",
    ):
        """Initialize a material object."""
        self._validate(name, eps_r, mu_r, sigma_e, sigma_m)
        self._name = name
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma_e = sigma_e
        self.sigma_m = sigma_m
        self.color = color

        self._register(self)

    @property
    def name(self):
        return self._name

    @staticmethod
    def _validate(name, eps_r=1, mu_r=1, sigma_e=0, sigma_m=0, color="#000000"):
        if not isinstance(name, str) or not name:
            raise ValueError("Material name must be a nonempty string.")
        for label, value in (
            ("eps_r", eps_r),
            ("mu_r", mu_r),
            ("sigma_e", sigma_e),
            ("sigma_m", sigma_m),
        ):
            if not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and nonnegative.")
            if label in ("eps_r", "mu_r") and value == 0:
                raise ValueError(f"{label} must be positive.")

    def _register(self, material: Material) -> None:
        """Register material."""
        materials_set = self.__class__.__dict__["__materials__"]
        if material.name not in materials_set:
            materials_set[material.name] = material
        else:
            raise MaterialExists(f"Material {material.name} already exists.")

    def _unregister(self, material):
        """Unregister material."""
        materials_set = self.__class__.__dict__["__materials__"]
        if materials_set.get(material.name) is material:
            del materials_set[material.name]

    def _delete(self) -> None:
        """Delete material."""
        self._unregister(self)

    @classmethod
    def load(cls, file_name: str):
        """Load materials from a .json file.

        Parameters
        ----------
        file_name : str
            Filename in the root directory.
        """
        with open(file_name, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                "Material file must contain a list of material definitions."
            )
        names = set()
        pending = []
        for props in data:
            if not isinstance(props, dict):
                raise ValueError("Each material definition must be an object.")
            cls._validate(**props)
            name = props["name"]
            if name in names:
                raise MaterialExists(f"Duplicate material {name} in file.")
            names.add(name)
            existing = cls.__dict__["__materials__"].get(name)
            if existing is None:
                pending.append(props)
            else:
                expected = {
                    "eps_r": 1,
                    "mu_r": 1,
                    "sigma_e": 0,
                    "sigma_m": 0,
                    "color": "#000000",
                }
                expected.update(props)
                if any(
                    getattr(existing, key) != value for key, value in expected.items()
                ):
                    raise MaterialExists(f"Conflicting material {name}.")
        for props in pending:
            cls(**props)
        return [cls[props["name"]] for props in data]

    @classmethod
    def from_name(cls, name: str) -> Material:
        """Initialize material from a name.

        Parameters
        ----------
        name : str
            Name of material.

        Returns
        -------
        Material

        Raises
        -------
        MaterialNotFound
            If material is not found.

        Notes
        -------
        Material[name] is equivalent to Material.from_name(name).

        """
        return cls[name]

    def __hash__(self) -> int:
        """Implement hash function."""
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """Implement equal function."""
        if not isinstance(other, Material):
            return NotImplemented
        return self.name == other.name

    def __repr__(self) -> str:
        """Representation of the material."""
        return f"Material({self.name})"
