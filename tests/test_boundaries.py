import numpy as np
import pytest

from fdtd import Grid, PeriodicBlochBoundary, PeriodicBoundary
from fdtd.utils import curl_E, curl_H


@pytest.mark.parametrize("beta", [0, 137.0])
def test_wrapped_derivatives_follow_discrete_bloch_mode(beta):
    grid = Grid((12, 1, 1), 1e-3)
    grid.add(PeriodicBlochBoundary((beta, 0, 0), x_direction=True))
    grid._prepare()
    k = 2 * np.pi / grid.x_max + beta
    mode = np.exp(-1j * k * grid.x)
    grid.E[:, 0, 0, 2] = mode
    grid.H[:, 0, 0, 2] = mode
    forward = curl_E(grid.E, *grid.spacing, grid._periodic, grid._phases)
    backward = curl_H(grid.H, *grid.spacing, grid._periodic, grid._phases)
    np.testing.assert_allclose(
        forward[:, 0, 0, 1],
        -(np.exp(-1j * k * grid.dx) - 1) / grid.dx * mode,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        backward[:, 0, 0, 1],
        -(1 - np.exp(1j * k * grid.dx)) / grid.dx * mode,
        atol=2e-12,
    )


def test_periodic_mode_remains_stable_and_does_not_duplicate_endpoint():
    grid = Grid((32, 1, 1), 1e-3)
    grid.add(PeriodicBoundary(x_direction=True))
    mode = np.exp(2j * np.pi * grid.x / grid.x_max)
    grid.E[:, 0, 0, 2] = mode
    grid.run(400, progress=False)
    assert np.max(np.abs(grid.E)) < 1.5
    assert abs(grid.E[0, 0, 0, 2] - grid.E[-1, 0, 0, 2]) > 0.01
