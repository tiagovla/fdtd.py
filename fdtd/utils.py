"""This module implements the utilities."""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .exceptions import WrongBounding


class Direction(Enum):
    """Direction of the object."""

    X = 0
    Y = 1
    Z = 2


@dataclass
class BoundingBox:
    """Implement a bounding box that contains an object."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        """Check bounding constraints."""
        if (self.x_max < self.x_min or self.y_max < self.y_min
                or self.z_max < self.z_min):
            raise WrongBounding


def curl_H(H: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """curl_H.

    Parameters
    ----------
    H : np.ndarray
        H field.
    dx : float
        Discrete spacing along the x axis.
    dy : float
        Discrete spacing along the y axis.
    dz : float
        Discrete spacing along the z axis.

    Returns
    -------
    curl : np.ndarray
        Return the curl of H.

    """
    curl = np.zeros_like(H)
    curl[:, 1:, :, 0] += (H[:, 1:, :, 2] - H[:, :-1, :, 2]) / dy
    curl[:, :, 1:, 0] -= (H[:, :, 1:, 1] - H[:, :, :-1, 1]) / dz

    curl[:, :, 1:, 1] += (H[:, :, 1:, 0] - H[:, :, :-1, 0]) / dz
    curl[1:, :, :, 1] -= (H[1:, :, :, 2] - H[:-1, :, :, 2]) / dx

    curl[1:, :, :, 2] += (H[1:, :, :, 1] - H[:-1, :, :, 1]) / dx
    curl[:, 1:, :, 2] -= (H[:, 1:, :, 0] - H[:, :-1, :, 0]) / dy

    return curl


def curl_E(E: np.ndarray, dx: float, dy: float, dz: float):
    """curl_E.

    Parameters
    ----------
    E : np.ndarray
        E field.
    dx : float
        Discrete spacing along the x axis.
    dy : float
        Discrete spacing along the y axis.
    dz : float
        Discrete spacing along the z axis.

    Returns
    -------
    curl : np.ndarray
        Return the curl of E.

    """
    curl = np.zeros_like(E)
    curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2]) / dy
    curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1]) / dz

    curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0]) / dz
    curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2]) / dx

    curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1]) / dx
    curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0]) / dy

    return curl
