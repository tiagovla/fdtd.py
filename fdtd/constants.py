"""This module contains constants."""
from math import pi

PI: float = pi
SPEED_LIGHT: float = 299_792_458.0
FREESPACE_PERMEABILITY: float = 4e-7 * pi
FREESPACE_PERMITTIVITY: float = 1.0 / (FREESPACE_PERMEABILITY * SPEED_LIGHT**2)
