import numpy as np
import pytest

from fdtd import Capacitor, Grid, ImpressedElectricCurrentSource, Inductor
from fdtd import PeriodicBlochBoundary, PeriodicBoundary, Resistor, VoltageDetector
from fdtd.constants import FREESPACE_PERMITTIVITY as EPS_0, SPEED_LIGHT
from fdtd.utils import Direction


@pytest.mark.parametrize("eps_r, beta", [(1, 0), (4, 173.0)])
def test_discrete_plane_wave_follows_dispersion_relation(eps_r, beta):
    grid = Grid((32, 1, 1), 1e-3, courant_factor=0.8)
    grid.cell_material[:] = (eps_r, 1, 0, 0)
    grid.add(PeriodicBlochBoundary((beta, 0, 0), x_direction=True))
    k = 2*np.pi/grid.x_max + beta
    mode = np.exp(-1j*k*grid.x)
    grid.E[:, 0, 0, 2] = mode
    theta = 2*np.arcsin(SPEED_LIGHT*grid.dt*np.sin(k*grid.dx/2)/(np.sqrt(eps_r)*grid.dx))
    grid.run(200, progress=False)
    amplitude = np.cos(200.5*theta)/np.cos(0.5*theta)
    np.testing.assert_allclose(grid.E[:, 0, 0, 2], amplitude*mode, atol=2e-13)


def test_measured_wave_speed_converges_with_grid_refinement():
    errors = []
    for cells in (32, 64, 128):
        grid = Grid((cells, 1, 1), 1/cells, courant_factor=0.6)
        grid.add(PeriodicBoundary(x_direction=True))
        grid.E[:, 0, 0, 2] = np.cos(2*np.pi*grid.x)
        grid._prepare()
        values = []
        for _ in range(cells):
            grid.step()
            values.append(grid.E[0, 0, 0, 2].real)
        values = np.array(values)
        cosine = np.dot(values[1:-1], values[:-2]+values[2:]) / (2*np.dot(values[1:-1], values[1:-1]))
        measured = np.arccos(cosine)/grid.dt
        errors.append(abs(measured/(2*np.pi*SPEED_LIGHT)-1))
    assert errors[0]/errors[1] > 3.9
    assert errors[1]/errors[2] > 3.9
    assert errors[-1] < 1e-4


def voltage_probe(grid, end):
    detector = VoltageDetector(0, 0, 0, end, 0, 0, direction=Direction.X)
    grid.add(detector)
    return detector


def test_resistive_terminal_discharge_matches_rc_decay():
    grid = Grid((6, 1, 1), 1e-3, courant_factor=0.3)
    grid.cell_material[:] = (2.5, 1, 0, 0)
    grid.add(Resistor(0, 0, 0, 4e-3, 0, 0, 5e4, direction=Direction.X))
    detector = voltage_probe(grid, 4e-3)
    grid.E[..., 0] = -1/4e-3
    grid.run(100, progress=False)
    capacitance = 2.5*EPS_0*1e-6/4e-3
    expected = np.exp(-detector.time/(5e4*capacitance))
    np.testing.assert_allclose(detector.captured.real, expected, rtol=1e-6)


def test_capacitive_terminal_charges_at_current_over_capacitance():
    grid = Grid((6, 1, 1), 1e-3, courant_factor=0.3)
    grid.cell_material[:] = (1, 1, 0, 0)
    grid.add(Capacitor(0, 0, 0, 3e-3, 0, 0, 1e-12, direction=Direction.X))
    grid.add(ImpressedElectricCurrentSource(0, 0, 0, 3e-3, 0, 0,
                                            waveform=lambda t: 0.01, direction=Direction.X))
    detector = voltage_probe(grid, 3e-3)
    grid.run(100, progress=False)
    total_capacitance = 1e-12 + EPS_0*1e-6/3e-3
    np.testing.assert_allclose(detector.captured.real,
                               0.01*1e-6*detector.time/total_capacitance, rtol=1e-13)


def test_inductive_terminal_follows_discrete_lc_oscillation_across_runs():
    grid = Grid((6, 1, 1), 1e-3, courant_factor=0.3)
    grid.cell_material[:] = (1, 1, 0, 0)
    grid.add(Inductor(0, 0, 0, 3e-3, 0, 0, inductance=1e-8, direction=Direction.X))
    detector = voltage_probe(grid, 3e-3)
    grid.E[..., 0] = -1/3e-3
    grid.run(70, progress=False)
    grid.run(130, progress=False)
    capacitance = EPS_0*1e-6/3e-3
    theta = 2*np.arcsin(grid.dt/(2*np.sqrt(1e-8*capacitance)))
    expected = np.cos((np.arange(1, 201)-0.5)*theta)/np.cos(theta/2)
    np.testing.assert_allclose(detector.captured.real, expected, atol=2e-14)
