import numpy as np
import pytest

from fdtd import CPMLBoundary, Grid, PeriodicBlochBoundary, PeriodicBoundary
from fdtd.constants import FREESPACE_PERMEABILITY as MU_0
from fdtd.constants import FREESPACE_PERMITTIVITY as EPS_0
from fdtd.constants import SPEED_LIGHT
from fdtd.utils import curl_H


def field_energy(grid):
    return np.sum(
        grid.eps_r * np.abs(grid.E) ** 2
        + MU_0 / EPS_0 * grid.mu_r * np.abs(grid.H) ** 2
    )


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("polarization", (1, 2))
@pytest.mark.parametrize("direction", (-1, 1))
@pytest.mark.parametrize("eps_r", (1, 4))
def test_outgoing_pulse_has_small_reflection_on_each_face(
    axis, polarization, direction, eps_r
):
    reflections = []
    for absorbing in (False, True):
        shape = [1, 1, 1]
        shape[axis] = 160
        grid = Grid(tuple(shape), 1e-3, courant_factor=0.8)
        grid.cell_material[:] = (eps_r, 1, 0, 0)
        if absorbing:
            grid.add(CPMLBoundary(12))
        e_component = (axis + polarization) % 3
        h_component = 3 - axis - e_component
        coordinates = np.arange(160) * grid.spacing[axis]
        transverse_shape = [1, 1, 1]
        transverse_shape[axis] = 160
        profile = np.exp(-(((coordinates - 0.08) / 0.008) ** 2)) * (1 + 0.3j)
        grid.E[..., e_component] = profile.reshape(transverse_shape)
        propagation = np.eye(3)[axis] * direction
        h_sign = np.cross(propagation, np.eye(3)[e_component])[h_component]
        h_profile = np.exp(
            -(
                (
                    (
                        coordinates
                        + grid.spacing[axis] / 2
                        + direction * SPEED_LIGHT / np.sqrt(eps_r) * grid.dt / 2
                        - 0.08
                    )
                    / 0.008
                )
                ** 2
            )
        )
        grid.H[..., h_component] = (
            h_sign * h_profile * (1 + 0.3j) / np.sqrt(MU_0 / (EPS_0 * eps_r))
        ).reshape(transverse_shape)
        grid._prepare()
        probe = [0, 0, 0]
        probe[axis] = 80
        trace = []
        for _ in range(380 * int(np.sqrt(eps_r))):
            grid.step()
            trace.append(abs(grid.E[tuple(probe) + (e_component,)]))
        reflections.append(max(trace[90 * int(np.sqrt(eps_r)) :]))
        assert np.all(np.isfinite(grid.E))
        assert np.all(np.isfinite(grid.H))
    assert reflections[0] > 0.9
    assert reflections[1] / reflections[0] < 1e-5


@pytest.mark.parametrize("magnetic", (False, True))
def test_2d_pulse_exits_through_edges_and_corners(magnetic):
    ratios = []
    for absorbing in (False, True):
        grid = Grid((60, 52, 1), (1e-3, 0.8e-3, 1.2e-3))
        if absorbing:
            grid.add(CPMLBoundary(10))
        x, y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
        profile = np.exp(
            -(((x - grid.x[-1] / 2) / (3 * grid.dx)) ** 2)
            - ((y - grid.y[-1] / 2) / (3 * grid.dy)) ** 2
        ) * (1 + 0.3j)
        if magnetic:
            grid.H[..., 2] = profile / np.sqrt(MU_0 / EPS_0)
        else:
            grid.E[..., 2] = profile
        grid._prepare()
        initial = field_energy(grid)
        grid.run(600, progress=False)
        ratios.append(field_energy(grid) / initial)
    assert ratios[0] > 0.9
    assert ratios[1] < 1e-6


def test_3d_transverse_pulse_is_absorbed_on_rectangular_grid():
    grid = Grid((26, 28, 30), (1e-3, 0.8e-3, 1.2e-3))
    grid.add(CPMLBoundary(6))
    coordinates = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    radius = sum(
        ((coordinate - (size - 1) * spacing / 2) / (3 * spacing)) ** 2
        for coordinate, size, spacing in zip(coordinates, grid.shape, grid.spacing)
    )
    potential = np.zeros_like(grid.E)
    potential[..., 2] = np.exp(-radius) * (1 + 0.3j)
    grid.E[:] = curl_H(potential, *grid.spacing)
    grid.E[:] /= np.max(np.abs(grid.E))
    grid._prepare()
    initial = field_energy(grid)
    grid.run(350, progress=False)
    assert field_energy(grid) / initial < 5e-5
    grid.run(350, progress=False)
    assert field_energy(grid) / initial < 5e-5


def test_oblique_bloch_wave_is_absorbed_with_periodic_transverse_axis():
    grid = Grid((120, 12, 1), 1e-3)
    grid.add(PeriodicBlochBoundary((0, 600, 0), y_direction=True))
    grid.add(CPMLBoundary(12, faces=("x-", "x+")))
    x, y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    grid.E[..., 2] = np.exp(-(((x - 0.06) / 0.01) ** 2)) * np.exp(
        -1j * (600 * x + 600 * y)
    )
    grid._prepare()
    initial = field_energy(grid)
    grid.run(800, progress=False)
    assert field_energy(grid) / initial < 1e-6


def test_cpml_preserves_continuation_and_clears_memory_on_reset():
    def simulation():
        grid = Grid((64, 1, 1), 1e-3)
        grid.add(CPMLBoundary(8))
        grid.E[:, 0, 0, 2] = np.exp(-(((grid.x - 0.032) / 0.004) ** 2))
        return grid

    split = simulation()
    split.run(70, progress=False)
    split.run(130, progress=False)
    whole = simulation()
    whole.run(200, progress=False)
    np.testing.assert_array_equal(split.E, whole.E)
    np.testing.assert_array_equal(split.H, whole.H)
    split.reset()
    split.E[:] = simulation().E
    split.run(200, progress=False)
    np.testing.assert_array_equal(split.E, whole.E)
    np.testing.assert_array_equal(split.H, whole.H)
    split.run(20, restart=True, progress=False)
    assert not np.any(split.E)
    assert not np.any(split.H)


def test_separate_face_boundaries_match_combined_boundary_and_can_be_removed():
    grids = [Grid((40, 1, 1), 1e-3) for _ in range(2)]
    grids[0].add(CPMLBoundary(8))
    left, right = CPMLBoundary(8, faces="x-"), CPMLBoundary(8, faces="x+")
    grids[1].add(left)
    grids[1].add(right)
    for grid in grids:
        grid.E[20, 0, 0, 2] = 1 + 2j
        grid.run(100, progress=False)
    np.testing.assert_array_equal(grids[0].E, grids[1].E)
    np.testing.assert_array_equal(grids[0].H, grids[1].H)
    grids[1].reset()
    left.detach_from_grid()
    right.detach_from_grid()
    grids[1].add(PeriodicBoundary(x_direction=True))
    grids[1].run(10, progress=False)
    assert not grids[1]._cpml_faces


def test_memory_estimate_includes_only_cpml_slabs_and_profiles():
    grid = Grid((24, 26, 28), 1e-3)
    baseline = grid.estimate_memory(10)
    boundary = CPMLBoundary(5)
    grid.add(boundary)
    estimate = boundary.estimate_memory()
    assert grid.estimate_memory(10) == baseline + estimate
    grid._prepare()
    allocated = sum(
        layer.psi_e.nbytes
        + layer.psi_h.nbytes
        + sum(array.nbytes for array in layer.coefficients_e + layer.coefficients_h)
        for layer in boundary._layers
    )
    assert estimate == allocated
    assert estimate < 4 * grid.E.nbytes


def test_zero_alpha_with_underflowed_grading_stays_finite():
    grid = Grid((32, 1, 1), 1e-3)
    grid.add(CPMLBoundary(4, order=1000, kappa_max=1, alpha_max=0))
    grid.E[16, 0, 0, 2] = 1
    with np.errstate(invalid="raise", divide="raise"):
        grid.run(100, progress=False)
    assert np.all(np.isfinite(grid.E))
    assert np.all(np.isfinite(grid.H))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thickness": 0},
        {"thickness": 1.5},
        {"thickness": True},
        {"order": 0},
        {"sigma_factor": 0},
        {"kappa_max": 0.5},
        {"alpha_min": -1},
        {"alpha_min": 0.1, "alpha_max": 0.05},
        {"alpha_max": np.nan},
        {"order": np.inf},
        {"sigma_factor": 1j},
        {"kappa_max": True},
        {"faces": ()},
        {"faces": 3},
        {"faces": (None,)},
        {"faces": ("x-", "x-")},
        {"faces": ("invalid",)},
    ],
)
def test_invalid_cpml_parameters(kwargs):
    with pytest.raises(ValueError):
        CPMLBoundary(**kwargs)


@pytest.mark.parametrize("periodic_first", (False, True))
def test_cpml_rejects_periodic_boundary_on_same_axis(periodic_first):
    grid = Grid((32, 1, 1), 1e-3)
    boundaries = [CPMLBoundary(8), PeriodicBoundary(x_direction=True)]
    for boundary in boundaries[:: -1 if periodic_first else 1]:
        grid.add(boundary)
    with pytest.raises(ValueError, match="share an axis"):
        grid.run(1, progress=False)
    assert grid.current_time_step == 0


@pytest.mark.parametrize(
    "boundaries, message",
    [
        ([CPMLBoundary(3, faces="y-")], "active"),
        ([CPMLBoundary(16)], "interior"),
        ([CPMLBoundary(31, faces="x+")], "interior"),
        ([CPMLBoundary(8, faces="x-"), CPMLBoundary(8, faces="x-")], "each face"),
        ([CPMLBoundary(15, faces="x-"), CPMLBoundary(16, faces="x+")], "interior"),
    ],
)
def test_invalid_cpml_layout(boundaries, message):
    grid = Grid((32, 1, 1), 1e-3)
    for boundary in boundaries:
        grid.add(boundary)
    with pytest.raises(ValueError, match=message):
        grid.run(1, progress=False)
    assert grid.current_time_step == 0
