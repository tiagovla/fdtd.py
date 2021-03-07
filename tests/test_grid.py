"""Test grid.py module."""
import numpy as np
import pytest
from fdtd import Grid


def test_courant_factor():
    with pytest.raises(ValueError):
        grid = Grid(shape=(10, 10, 10), spacing=1e-3, courant_factor=3)

    with pytest.raises(ValueError):
        grid = Grid(shape=(10, 10, 10), spacing=1e-3, courant_factor=0)


def test_time_step():
    # 3d:
    grid = Grid(shape=(10, 10, 10), spacing=1e-3, courant_factor=0.8)
    assert grid.dt == pytest.approx(1.54e-12, rel=1e-10)
    # 2d:
    grid = Grid(shape=(10, 10, 1), spacing=2e-3, courant_factor=0.8)
    assert grid.dt == pytest.approx(3.77e-12, rel=1e-10)
    # 1d:
    grid = Grid(shape=(10, 1, 1), spacing=3e-3, courant_factor=0.8)
    assert grid.dt == pytest.approx(8.00e-12, rel=1e-10)


def test_grid_shape():
    grid = Grid(shape=(10, 10, 10), spacing=1e-3)
    assert grid.shape == (10, 10, 10)
    assert grid.E.shape == grid.shape + (3, )
    assert grid.H.shape == grid.shape + (3, )
    assert grid.c_ee.shape == grid.shape + (3, )
    assert grid.c_eh.shape == grid.shape + (3, )
    assert grid.c_he.shape == grid.shape + (3, )
    assert grid.c_hh.shape == grid.shape + (3, )
    assert grid.cell_material.shape == grid.shape + (4, )


def test_grid_calculate_material_simple():
    grid = Grid(shape=(2, 2, 2), spacing=1e-3)
    grid.cell_material = np.array([
        [
            [[2.0, 1.0, 3.0, 0.0], [0.0, 0.0, 4.0, 0.0]],
            [[8.0, 2.0, 0.0, 0.0], [0.4, 4.0, 0.0, 0.0]],
        ],
        [
            [[2.0, 0.0, 0.4, 0.0], [0.0, 0.0, 2.0, 0.0]],
            [[0.0, 8.0, 0.0, 9.0], [0.0, 9.0, 2.0, 1.0]],
        ],
    ])
    grid._calculate_material_components_simple()
    for prop, mat in zip(
        [grid.eps_r, grid.mu_r, grid.sigma_e, grid.sigma_m],
        [grid.cell_material[:, :, :, k] for k in range(4)],
    ):
        for dim in range(3):
            assert prop[:, :, :, dim] == pytest.approx(mat, rel=0.01)
