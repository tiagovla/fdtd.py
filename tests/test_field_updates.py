"""Regression checks for staggered field updates in lossy media."""

import json
from pathlib import Path

import numpy as np

from fdtd import Grid


def reference_data():
    return json.loads((Path(__file__).parent / "reference/fixtures.json").read_text())


def test_lossy_interior_update_matches_saved_solution():
    case = reference_data()["core"]
    shape = tuple(case["shape"])
    grid = Grid(shape, tuple(case["spacing"]), courant_factor=0.7)
    grid.cell_material[:] = case["medium"]
    grid._prepare()
    grid.E[:] = np.asarray(case["initial_E"]).reshape(shape + (3,), order="F")
    grid.H[:] = np.asarray(case["initial_H"]).reshape(shape + (3,), order="F")
    grid.step()
    for field in ("E", "H"):
        expected = np.asarray(case["final_" + field]).reshape(shape + (3,), order="F")
        np.testing.assert_allclose(
            getattr(grid, field)[1:-1, 1:-1, 1:-1],
            expected[1:-1, 1:-1, 1:-1],
            rtol=2e-13,
            atol=1e-15,
        )
