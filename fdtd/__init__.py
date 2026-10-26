__version__ = "0.0.3"

import logging

from fdtd.boundaries import PeriodicBlochBoundary, PeriodicBoundary
from fdtd.detectors import EFieldDetector, HFieldDetector, VoltageDetector
from fdtd.grid import Grid
from fdtd.lumped_elements import Capacitor, Diode, Inductor, Resistor
from fdtd.materials import Material
from fdtd.objects import Brick, Sphere
from fdtd.pml import CPMLBoundary
from fdtd.sources import (
    EFieldSource,
    ImpressedElectricCurrentSource,
    ImpressedMagneticCurrentSource,
    VoltageSource,
)
from fdtd.waveforms import GaussianWaveform, SineWaveform, StepWaveform

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "Brick",
    "Capacitor",
    "CPMLBoundary",
    "Diode",
    "EFieldDetector",
    "EFieldSource",
    "GaussianWaveform",
    "Grid",
    "HFieldDetector",
    "ImpressedElectricCurrentSource",
    "ImpressedMagneticCurrentSource",
    "Inductor",
    "Material",
    "PeriodicBlochBoundary",
    "PeriodicBoundary",
    "Resistor",
    "SineWaveform",
    "Sphere",
    "StepWaveform",
    "VoltageDetector",
    "VoltageSource",
]
