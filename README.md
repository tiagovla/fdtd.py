[![PyPI license](https://img.shields.io/pypi/l/fdtd.py.svg)](https://pypi.python.org/pypi/fdtd.py/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/fdtd.py.svg)](https://pypi.python.org/pypi/fdtd.py/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/fdtd.py.svg)](https://pypi.python.org/pypi/fdtd.py/)
[![Build Status](https://travis-ci.com/tiagovla/fdtd.py.svg?branch=master)](https://travis-ci.com/tiagovla/fdtd.py)
[![DeepSource](https://deepsource.io/gh/tiagovla/fdtd.py.svg/?label=active+issues)](https://deepsource.io/gh/tiagovla/fdtd.py/?ref=repository-badge)
[![codecov](https://codecov.io/gh/tiagovla/fdtd.py/branch/master/graph/badge.svg?token=MC1GNINTAY)](https://codecov.io/gh/tiagovla/fdtd.py)
[![Documentation Status](https://readthedocs.org/projects/fdtd-py/badge/?version=v0.0.3)](https://fdtd-py.readthedocs.io/en/v0.0.3/?badge=v0.0.3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4648395.svg)](https://doi.org/10.5281/zenodo.4648395)

# FDTD Framework for Electromagnetic Simulations

## Installation:
```bash
pip install fdtd.py
```

## Example:
```python

"""Voltage source + resistor example."""

from fdtd import Brick, Grid, Material, Resistor, SineWaveform
from fdtd import VoltageSource as VSource
from fdtd import VoltageDetector as VDetector

print("Loading materials...")
Material.load("materials.json")

print("Creating grid...")
grid = Grid(shape=(10, 10, 10), spacing=1e-3, courant_factor=0.9)

print("Creating objects...")
brick1 = Brick(*(4e-3, 2e-3, 8e-3), *(6e-3, 8e-3, 8e-3), "PEC")
brick2 = Brick(*(4e-3, 2e-3, 4e-3), *(6e-3, 8e-3, 4e-3), "PEC")
v_source = VSource(*(4e-3, 2e-3, 4e-3),
                   *(6e-3, 2e-3, 8e-3),
                   waveform=SineWaveform(frequency=1e9),
                   resistance=50)
resistor = Resistor(*(4e-3, 8e-3, 4e-3), *(6e-3, 8e-3, 8e-3), 50)
v_detector = VDetector(*(4e-3, 5e-3, 4e-3),
                             *(6e-3, 5e-3, 8e-3),
                             plot=True)

print("Adding components to grid...")
components = [brick1, brick2, v_source, resistor, v_detector]
for comp in components:
    grid.add(comp)

print("Running simulation...")
grid.run(n_steps=5000)
grid.plot_3d()

```


### References
[1] A. Z. Elsherbeni and V. Demir, The finite-difference time-domain: method for electromagnetics with MATLAB simulations, 2nd Edition. Edison, NJ: SciTech Publishing, an imprint of the IET, 2016.
