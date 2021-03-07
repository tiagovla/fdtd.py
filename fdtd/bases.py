"""This module contains base classes."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

from .exceptions import AlreadyRegistered

if TYPE_CHECKING:
    from .grid import Grid


class FDTDElementBase(ABC):
    """Base for every FDTD element."""

    quantity = 0

    def __init__(self):
        """Initialize the object."""
        self.grid: Optional[Grid] = None

    def _register_grid(self, grid: Grid):
        """Register grid to the object."""
        if self.grid is None:
            self.grid = grid
        else:
            raise AlreadyRegistered("Object already registered to the grid.")

    @classmethod
    def _create_new_name(cls):
        """Create new name for the object."""
        cls.quantity += 1
        return f"{cls.__name__}_{cls.quantity}"

    def attach_to_grid(self):
        """Attach object to grid."""

    def detach_from_grid(self):
        """Detach from grid."""
        if self.grid:
            self.grid = None
