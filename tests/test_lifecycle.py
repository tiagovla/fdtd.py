import numpy as np

from fdtd import Brick, EFieldDetector, Grid, Material, VoltageSource


def simulation():
    grid = Grid((4, 4, 4), 1e-3)
    grid.add(VoltageSource(1e-3, 1e-3, 0, 1e-3, 1e-3, 2e-3, waveform=lambda time: 1.0))
    detector = EFieldDetector(1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3)
    grid.add(detector)
    return grid, detector


def test_split_run_matches_uninterrupted_run():
    grid, detector = simulation()
    grid.run(3, progress=False)
    grid.run(5, progress=False)
    expected, expected_detector = simulation()
    expected.run(8, progress=False)
    np.testing.assert_allclose(grid.E, expected.E)
    np.testing.assert_allclose(grid.H, expected.H)
    np.testing.assert_allclose(detector.captured, expected_detector.captured)
    assert grid.current_time_step == 8


def test_reset_and_restart_match_fresh_grid():
    grid, detector = simulation()
    grid.run(8, progress=False)
    first = detector.captured.copy()
    grid.reset()
    assert not np.any(grid.E)
    assert not np.any(grid.H)
    assert detector.captured is None
    grid.run(8, progress=False)
    np.testing.assert_allclose(detector.captured, first)
    grid.run(8, restart=True, progress=False)
    np.testing.assert_allclose(detector.captured, first)


def test_reset_and_object_removal_restore_background_material():
    grid = Grid((4, 4, 4), 1e-3)
    grid.cell_material[:] = (2, 1, 0, 0)
    material = Material("reset-object-background", eps_r=6)
    brick = Brick(0, 0, 0, 2e-3, 2e-3, 2e-3, material)
    grid.add(brick)
    grid.run(1, progress=False)
    assert grid.cell_material[0, 0, 0, 0] == 6
    grid.reset()
    brick.detach_from_grid()
    grid.cell_material[0, 0, 0, 0] = 3
    grid.run(1, progress=False)
    assert grid.cell_material[0, 0, 0, 0] == 3
    assert grid.cell_material[1, 1, 1, 0] == 2
