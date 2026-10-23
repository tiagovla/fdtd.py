import numpy as np
import pytest

from fdtd import Grid, Material, Sphere


@pytest.mark.parametrize("shape", [(4, 5, 3), (4, 5, 1), (4, 1, 1)])
def test_uniform_material_averages_preserve_values_and_zero_conductivity(shape):
    grid = Grid(shape, 1e-3)
    grid.cell_material[:] = [3.0, 2.0, 0.0, 0.0]
    grid._prepare()
    for field, expected in [
        (grid.eps_r, 3),
        (grid.mu_r, 2),
        (grid.sigma_e, 0),
        (grid.sigma_m, 0),
    ]:
        np.testing.assert_allclose(field, expected)


def test_material_interface_uses_arithmetic_and_harmonic_means():
    grid = Grid((4, 4, 4), 1e-3)
    grid.cell_material[:] = [2, 2, 0.01, 0.03]
    grid.cell_material[2:] = [6, 6, 0.05, 0.09]
    grid._prepare()
    assert grid.eps_r[2, 2, 2, 1] == pytest.approx(4)
    assert grid.eps_r[2, 2, 2, 0] == pytest.approx(6)
    assert grid.mu_r[2, 2, 2, 0] == pytest.approx(3)
    assert grid.sigma_e[2, 2, 2, 1] == pytest.approx(0.03)
    assert grid.sigma_m[2, 2, 2, 0] == pytest.approx(0.045)


def test_sphere_rasterization_supports_unequal_axes():
    material = Material("unequal-axis-sphere", eps_r=3)
    grid = Grid((8, 5, 4), (1e-3, 2e-3, 3e-3))
    grid.add(Sphere(4e-3, 5e-3, 6e-3, 3e-3, material))
    grid._prepare()
    x, y, z = np.meshgrid(grid.x_c, grid.y_c, grid.z_c, indexing="ij")
    inside = (x - 4e-3) ** 2 + (y - 5e-3) ** 2 + (z - 6e-3) ** 2 <= (3e-3) ** 2
    np.testing.assert_array_equal(grid.cell_material[..., 0] == 3, inside)
