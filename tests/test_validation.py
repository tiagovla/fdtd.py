import numpy as np
import pytest

from fdtd import EFieldDetector, Grid, VoltageSource


@pytest.mark.parametrize("shape", [(1, 1, 1), (0, 3, 3), (2.5, 3, 3), (2, 3)])
def test_invalid_grid_shapes(shape):
    with pytest.raises(ValueError):
        Grid(shape, 1e-3)


@pytest.mark.parametrize("spacing", [0, -1, np.inf, np.nan, (1, 0, 1), (1, 2)])
def test_invalid_grid_spacing(spacing):
    with pytest.raises(ValueError):
        Grid((3, 3, 3), spacing)


def test_integer_scalar_spacing_is_supported():
    assert Grid((3, 3, 3), 1).spacing == (1, 1, 1)


@pytest.mark.parametrize("kwargs", [{"n_steps": -1}, {"n_steps": 1.5},
                                   {"total_time": 0}, {"total_time": np.inf}])
def test_invalid_runs_do_not_advance_state(kwargs):
    grid = Grid((3, 3, 3), 1e-3)
    with pytest.raises(ValueError):
        grid.run(**kwargs)
    assert grid.current_time_step == 0


def test_outside_detector_is_rejected_without_registration():
    grid = Grid((3, 3, 3), 1e-3)
    detector = EFieldDetector(0, 0, 0, 1, 0, 0)
    with pytest.raises(ValueError, match="bounds"):
        grid.add(detector)
    assert detector.grid is None


def test_detach_removes_the_grid_list_entry_and_allows_reregistration():
    grid = Grid((3, 3, 3), 1e-3)
    detector = EFieldDetector(0, 0, 0, 0, 0, 0)
    grid.add(detector)
    detector.detach_from_grid()
    assert not grid.detectors
    grid.add(detector)
    grid.run(1, progress=False)
    with pytest.raises(RuntimeError, match="Reset"):
        detector.detach_from_grid()


def test_voltage_source_requires_a_nonzero_axial_extent():
    grid = Grid((3, 3, 3), 1e-3)
    grid.add(VoltageSource(0, 0, 0, 0, 0, 0, waveform=lambda t: 1))
    with pytest.raises(ValueError, match="edge"):
        grid.run(1, progress=False)
