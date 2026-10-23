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


def _add_periodic_curl(curl, field, spacing, periodic, phases, forward):
    if periodic is None:
        return curl
    phases = (1, 1, 1) if phases is None else phases
    for axis, component, target, sign in ((1, 2, 0, 1), (2, 1, 0, -1),
                                         (2, 0, 1, 1), (0, 2, 1, -1),
                                         (0, 1, 2, 1), (1, 0, 2, -1)):
        if not periodic[axis] or field.shape[axis] == 1:
            continue
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis], last[axis] = 0, -1
        first, last = tuple(first), tuple(last)
        if forward:
            difference = field[first + (component,)] * phases[axis] - field[last + (component,)]
            index = last
        else:
            difference = field[first + (component,)] - field[last + (component,)] / phases[axis]
            index = first
        curl[index + (target,)] += sign * difference / spacing[axis]
    return curl


def curl_H(H: np.ndarray, dx: float, dy: float, dz: float, periodic=None, phases=None) -> np.ndarray:
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
    curl = np.zeros_like(H, dtype=np.result_type(H.dtype, float))
    curl[:, 1:, :, 0] += (H[:, 1:, :, 2] - H[:, :-1, :, 2]) / dy
    curl[:, :, 1:, 0] -= (H[:, :, 1:, 1] - H[:, :, :-1, 1]) / dz

    curl[:, :, 1:, 1] += (H[:, :, 1:, 0] - H[:, :, :-1, 0]) / dz
    curl[1:, :, :, 1] -= (H[1:, :, :, 2] - H[:-1, :, :, 2]) / dx

    curl[1:, :, :, 2] += (H[1:, :, :, 1] - H[:-1, :, :, 1]) / dx
    curl[:, 1:, :, 2] -= (H[:, 1:, :, 0] - H[:, :-1, :, 0]) / dy

    return _add_periodic_curl(curl, H, (dx, dy, dz), periodic, phases, False)


def curl_E(E: np.ndarray, dx: float, dy: float, dz: float, periodic=None, phases=None):
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
    curl = np.zeros_like(E, dtype=np.result_type(E.dtype, float))
    curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2]) / dy
    curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1]) / dz

    curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0]) / dz
    curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2]) / dx

    curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1]) / dx
    curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0]) / dy

    return _add_periodic_curl(curl, E, (dx, dy, dz), periodic, phases, True)
