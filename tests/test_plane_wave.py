"""Physical checks for material-contrast and TF/SF plane-wave injection."""

import numpy as np
import pytest

from fdtd import (
    CPMLBoundary,
    GaussianWaveform,
    Grid,
    Material,
    PeriodicBlochBoundary,
    PeriodicBoundary,
    PlaneWave,
    Resistor,
    ScatteredFieldSource,
    TFSFSource,
)
from fdtd.constants import FREESPACE_PERMEABILITY as MU_0
from fdtd.constants import FREESPACE_PERMITTIVITY as EPS_0
from fdtd.constants import SPEED_LIGHT
from fdtd.utils import Direction, curl_E, curl_H


@pytest.fixture(scope="module")
def media():
    materials = {
        "vacuum": Material("plane-wave-test-vacuum"),
        "dielectric": Material("plane-wave-test-dielectric", eps_r=4),
        "magnetodielectric": Material(
            "plane-wave-test-magnetodielectric", eps_r=4, mu_r=2
        ),
        "contrast": Material("plane-wave-test-contrast", eps_r=1.3, mu_r=1.2),
        "fast": Material("plane-wave-test-fast", eps_r=0.01),
        "lossy_e": Material("plane-wave-test-lossy-e", sigma_e=0.1),
        "lossy_m": Material("plane-wave-test-lossy-m", sigma_m=0.1),
    }
    yield materials
    for material in materials.values():
        material._delete()


def incident_grid(grid, wave, magnetic, time):
    fields = np.empty_like(grid.E)
    nodes = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    for component in range(3):
        positions = np.stack(nodes, axis=-1)
        for axis, spacing in enumerate(grid.spacing):
            positions[..., axis] += (
                spacing / 2 * ((axis != component) if magnetic else axis == component)
            )
        evaluate = wave.magnetic_field if magnetic else wave.electric_field
        fields[..., component] = evaluate(positions, time)[..., component]
    return fields


def test_plane_wave_polarization_impedance_and_retarded_time(media):
    wave = PlaneWave(
        lambda t: 2 + 3j * t,
        material=media["magnetodielectric"],
        direction=(0, -2, 0),
        polarization=(1, 0, 1j),
        origin=(0, 4, 0),
    )
    assert wave.material is media["magnetodielectric"]
    assert wave.speed == pytest.approx(SPEED_LIGHT / np.sqrt(8))
    assert wave.impedance == pytest.approx(np.sqrt(2 * MU_0 / (4 * EPS_0)))
    positions = np.array([[0, 4, 0], [0, 4 - wave.speed, 0]])
    electric = wave.electric_field(positions, 2)
    magnetic = wave.magnetic_field(positions, 2)
    np.testing.assert_allclose(electric[:, 0], np.array([2 + 6j, 2 + 3j]) / np.sqrt(2))
    np.testing.assert_allclose(
        np.cross(wave.direction, electric) / wave.impedance, magnetic
    )
    np.testing.assert_allclose(electric @ wave.direction, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"waveform": None},
        {"direction": (0, 0, 0)},
        {"direction": (1j, 0, 0)},
        {"direction": (1, 0)},
        {"polarization": (0, 0, 0)},
        {"polarization": (1, 0, 1)},
        {"origin": (0, np.nan, 0)},
        {"material": None},
        {"material": "plane-wave-test-vacuum"},
        {"material": 1},
    ],
)
def test_invalid_plane_waves(kwargs, media):
    with pytest.raises(ValueError):
        PlaneWave(**{"waveform": lambda t: 1, "material": media["vacuum"], **kwargs})


@pytest.mark.parametrize("material", ("lossy_e", "lossy_m"))
def test_plane_wave_rejects_lossy_background_material(material, media):
    with pytest.raises(ValueError, match="lossless"):
        PlaneWave(lambda t: 1, material=media[material])


@pytest.mark.parametrize("conductivity", ("sigma_e", "sigma_m"))
def test_background_material_is_revalidated_at_source_preparation(conductivity, media):
    material = media["vacuum"]
    grid = Grid((16, 1, 1), 1e-3)
    grid.add(ScatteredFieldSource(PlaneWave(lambda t: 1, material=material)))
    try:
        setattr(material, conductivity, 0.1)
        with pytest.raises(ValueError, match="lossless"):
            grid.run(1, progress=False)
        assert grid.current_time_step == 0
    finally:
        setattr(material, conductivity, 0)


@pytest.mark.parametrize("waveform", [lambda t: np.nan, lambda t: np.ones((2, 3))])
def test_invalid_waveform_samples(waveform, media):
    with pytest.raises(ValueError, match="waveform"):
        PlaneWave(waveform, material=media["vacuum"]).electric_field((0, 0, 0), 0)


def test_scattered_update_is_total_constitutive_update_minus_incident_fields(media):
    grid = Grid((7, 8, 9), (1e-3, 0.8e-3, 1.2e-3))
    grid.cell_material[:] = (2.3, 1.7, 0.15, 0.25)
    wave = PlaneWave(
        lambda t: (1 + 0.3j) * (1 + t / (10 * grid.dt)),
        material=media["contrast"],
        direction=(1, 2, 3),
        polarization=(1, 1, -1),
    )
    grid.add(ScatteredFieldSource(wave))
    grid._prepare()
    electric_old = incident_grid(grid, wave, False, 0)
    magnetic_old = incident_grid(grid, wave, True, -grid.dt / 2)
    magnetic_incident = incident_grid(grid, wave, True, grid.dt / 2)
    electric_incident = incident_grid(grid, wave, False, grid.dt)
    # An affine traveling wave satisfies the background Yee derivatives exactly.
    magnetic_total = grid.c_hh * magnetic_old + grid.c_he * curl_E(
        electric_old, *grid.spacing
    )
    electric_total = grid.c_ee * electric_old + grid.c_eh * curl_H(
        magnetic_total, *grid.spacing
    )
    grid.step()
    np.testing.assert_allclose(
        grid.H[1:-1, 1:-1, 1:-1],
        (magnetic_total - magnetic_incident)[1:-1, 1:-1, 1:-1],
        atol=1e-17,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        grid.E[2:-2, 2:-2, 2:-2],
        (electric_total - electric_incident)[2:-2, 2:-2, 2:-2],
        atol=1e-14,
        rtol=2e-13,
    )


def test_homogeneous_background_has_no_scattered_field(media):
    grid = Grid((26, 22, 1), (1e-3, 0.8e-3, 1e-3))
    grid.cell_material[:] = (4, 2, 0, 0)
    grid.add(CPMLBoundary(4))
    wave = PlaneWave(
        lambda t: np.exp(1j * t / grid.dt), material=media["magnetodielectric"]
    )
    source = ScatteredFieldSource(wave)
    grid.add(source)
    grid.run(40, progress=False)
    assert not np.any(grid.E)
    assert not np.any(grid.H)
    assert source.estimate_memory() == 0


def test_tfsf_surface_matches_the_six_face_curl_jump(media):
    grid = Grid((11, 12, 13), (1e-3, 0.8e-3, 1.2e-3))
    wave = PlaneWave(
        lambda t: (1 + 0.2j) * (1 + t / (10 * grid.dt)),
        material=media["vacuum"],
        direction=(1, 2, 3),
        polarization=(1, 1, -1),
    )
    lower, upper = np.array((3, 3, 3)), np.array((8, 9, 10))
    source = TFSFSource(*(lower * grid.spacing), *(upper * grid.spacing), wave=wave)
    grid.add(source)
    grid._prepare()
    masks = []
    coordinates = np.indices(grid.shape).transpose(1, 2, 3, 0)
    for magnetic in (False, True):
        mask = np.empty_like(grid.E, dtype=bool)
        for component in range(3):
            offsets = (
                (np.arange(3) != component) if magnetic else (np.arange(3) == component)
            )
            positions = coordinates + offsets / 2
            mask[..., component] = np.all(
                (positions >= lower) & (positions < upper), axis=-1
            )
        masks.append(mask)
    electric = incident_grid(grid, wave, False, 0)
    magnetic = incident_grid(grid, wave, True, grid.dt / 2)
    expected_h = grid.c_he * (
        masks[1] * curl_E(electric, *grid.spacing)
        - curl_E(masks[0] * electric, *grid.spacing)
    )
    expected_e = grid.c_eh * (
        masks[0] * curl_H(magnetic, *grid.spacing)
        - curl_H(masks[1] * magnetic, *grid.spacing)
    )
    grid.current_time = grid.dt / 2
    source.update_H()
    grid.current_time = grid.dt
    source.update_E()
    np.testing.assert_allclose(grid.H, expected_h, atol=1e-17)
    np.testing.assert_allclose(grid.E, expected_e, atol=1e-14)
    assert not np.any(grid.E[4:7, 4:8, 4:9])
    assert not np.any(grid.H[4:7, 4:8, 4:9])


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize("polarization", (1, 2))
def test_tfsf_pulse_propagates_with_small_leakage_on_each_axis(
    axis, sign, polarization, media
):
    shape = [1, 1, 1]
    shape[axis] = 140
    spacing = (1e-3, 0.8e-3, 1.2e-3)
    grid = Grid(tuple(shape), spacing, courant_factor=0.8)
    background = 1 if axis == 0 else 4
    grid.cell_material[:] = (background, 1, 0, 0)
    grid.add(CPMLBoundary(10))
    direction = np.eye(3)[axis] * sign
    component = (axis + polarization) % 3
    origin = np.zeros(3)
    origin[axis] = (shape[axis] - 1) * spacing[axis] if sign < 0 else 0
    wave = PlaneWave(
        lambda t: (1 + 0.3j) * np.exp(-(((t - 5 * tau) / tau) ** 2)),
        material=media["vacuum" if background == 1 else "dielectric"],
        direction=direction,
        polarization=np.eye(3)[component],
        origin=origin,
    )
    tau = 12 * spacing[axis] / wave.speed
    lower, upper = np.zeros(3), np.zeros(3)
    lower[axis], upper[axis] = 25 * spacing[axis], 110 * spacing[axis]
    grid.add(TFSFSource(*lower, *upper, wave=wave))
    probe = [0, 0, 0]
    probe[axis] = 70
    position = np.array(probe) * spacing
    error, leakage, peak = 0, 0, 0
    steps = int(np.ceil((11 * tau + 140 * spacing[axis] / wave.speed) / grid.dt))
    for _ in range(steps):
        grid.step()
        value = grid.E[tuple(probe) + (component,)]
        expected = wave.electric_field(position, grid.time_passed)[component]
        error = max(error, abs(value - expected))
        peak = max(peak, abs(value))
        exterior = np.moveaxis(grid.E[..., component], axis, 0)
        leakage = max(leakage, np.max(abs(exterior[:25])), np.max(abs(exterior[110:])))
    assert peak > 1
    assert error < 0.01
    assert leakage < 0.01


@pytest.mark.parametrize(
    "shape,direction,polarization",
    [
        ((58, 54, 1), (3, 4, 0), (0, 0, 1)),
        ((58, 54, 1), (3, 4, 0), (-4, 3, 0)),
        ((24, 26, 28), (1, 2, -1), (2, -1, 0)),
    ],
)
def test_oblique_tfsf_pulse_is_confined_in_2d_and_3d(
    shape, direction, polarization, media
):
    spacing = np.array((1, 0.8, 1.2)) * 1e-3
    grid = Grid(shape, tuple(spacing), courant_factor=0.75)
    pml, lower_cells, upper_margin = (8, 10, 10) if shape[2] == 1 else (4, 6, 7)
    grid.add(CPMLBoundary(pml))
    origin = np.where(np.array(direction) < 0, (np.array(shape) - 1) * spacing, 0)
    wave = PlaneWave(
        lambda t: np.exp(-(((t - 5 * tau) / tau) ** 2)),
        material=media["vacuum"],
        direction=direction,
        polarization=polarization,
        origin=origin,
    )
    tau = 10e-3 / wave.speed
    lower = np.where(np.array(shape) > 1, lower_cells * spacing, 0)
    upper = np.where(np.array(shape) > 1, (np.array(shape) - upper_margin) * spacing, 0)
    source = TFSFSource(*lower, *upper, wave=wave)
    grid.add(source)
    grid._prepare()
    mask = source._mask(False)
    center = np.array(shape) // 2
    component = int(np.argmax(abs(wave.polarization)))
    position = center * spacing
    position[component] += spacing[component] / 2
    error, leakage, peak = 0, 0, 0
    for _ in range(300):
        grid.step()
        value = grid.E[(*center, component)]
        error = max(
            error,
            abs(value - wave.electric_field(position, grid.time_passed)[component]),
        )
        leakage = max(leakage, np.max(abs(grid.E[~mask])))
        peak = max(peak, abs(value))
    assert peak > 0.5
    assert error < 0.01
    assert leakage < 0.01


@pytest.mark.parametrize("eps_r,mu_r,reflection", [(4, 1, -1 / 3), (1, 4, 1 / 3)])
def test_both_formulations_agree_with_slab_fresnel_reflection(
    eps_r, mu_r, reflection, media
):
    traces = []
    for kind in (ScatteredFieldSource, TFSFSource):
        grid = Grid((260, 1, 1), 1e-3, courant_factor=0.8)
        grid.cell_material[:] = (1, 1, 0, 0)
        grid.cell_material[130:150, ..., :2] = (eps_r, mu_r)
        grid.add(CPMLBoundary(12))
        wave = PlaneWave(
            lambda t: np.exp(-(((t - 5 * tau) / tau) ** 2)), material=media["vacuum"]
        )
        tau = 12e-3 / wave.speed
        source = (
            kind(wave)
            if kind is ScatteredFieldSource
            else kind(0.04, 0, 0, 0.22, 0, 0, wave=wave)
        )
        grid.add(source)
        trace = []
        for _ in range(600):
            grid.step()
            trace.append(grid.E[25, 0, 0, 2])
        trace = np.asarray(trace)
        arrival = 5 * tau + (2 * 0.13 - 0.025) / wave.speed
        times = (np.arange(len(trace)) + 1) * grid.dt
        first_pulse = trace[abs(times - arrival) < 1.5 * tau].real
        value = np.min(first_pulse) if reflection < 0 else np.max(first_pulse)
        assert value == pytest.approx(reflection, abs=0.005)
        traces.append(trace)
    np.testing.assert_allclose(traces[0], traces[1], atol=0.003)


def test_oblique_scattering_formulations_converge_together_under_refinement(media):
    differences = []
    for refinement in (1, 2):
        traces = []
        for kind in (ScatteredFieldSource, TFSFSource):
            cells, spacing = 48 * refinement, 1e-3 / refinement
            grid = Grid((cells, cells, 1), spacing, courant_factor=0.8)
            grid.cell_material[:] = (1, 1, 0, 0)
            region = slice(20 * refinement, 28 * refinement)
            grid.cell_material[region, region, :, 0] = 4
            grid.add(CPMLBoundary(6 * refinement))
            wave = PlaneWave(
                GaussianWaveform(5 * 0.012 / SPEED_LIGHT, 0.012 / SPEED_LIGHT),
                material=media["vacuum"],
                direction=(1, 1, 0),
            )
            source = (
                kind(wave)
                if kind is ScatteredFieldSource
                else kind(0.01, 0.01, 0, 0.038, 0.038, 0, wave=wave)
            )
            grid.add(source)
            trace = []
            for _ in range(280 * refinement):
                grid.step()
                trace.append(grid.E[8 * refinement, 24 * refinement, 0, 2])
            assert np.max(abs(np.asarray(trace))) > 0.15
            traces.append(np.asarray(trace))
        differences.append(np.max(abs(traces[0] - traces[1])))
    assert differences[0] < 0.001
    assert differences[1] < differences[0] / 3.5


@pytest.mark.parametrize("kind", (ScatteredFieldSource, TFSFSource))
def test_plane_wave_continuation_reset_removal_and_memory(kind, media):
    def simulation():
        grid = Grid((96, 1, 1), 1e-3)
        grid.cell_material[50:60, ..., 0] = 3
        grid.add(CPMLBoundary(8))
        wave = PlaneWave(GaussianWaveform(1e-10, 2e-11), material=media["vacuum"])
        source = (
            kind(wave)
            if kind is ScatteredFieldSource
            else kind(0.02, 0, 0, 0.076, 0, 0, wave=wave)
        )
        grid.add(source)
        return grid, source

    whole, source = simulation()
    bound = source.estimate_memory()
    whole.run(200, progress=False)
    assert 0 < source.estimate_memory() <= bound
    allocated = sum(
        array.nbytes
        for terms in (source._electric_terms, source._magnetic_terms)
        for term in terms
        for array in term
    )
    assert source.estimate_memory() == allocated
    without_cache = source._electric_terms, source._magnetic_terms
    source._electric_terms, source._magnetic_terms = [], []
    baseline = whole.estimate_memory(200)
    source._electric_terms, source._magnetic_terms = without_cache
    assert whole.estimate_memory(200) == baseline + allocated
    split, _ = simulation()
    split.run(63, progress=False)
    split.run(137, progress=False)
    np.testing.assert_array_equal(split.E, whole.E)
    np.testing.assert_array_equal(split.H, whole.H)
    split.run(200, restart=True, progress=False)
    np.testing.assert_array_equal(split.E, whole.E)
    np.testing.assert_array_equal(split.H, whole.H)
    split.reset()
    split.sources[0].detach_from_grid()
    assert not split.sources
    split.run(10, progress=False)
    assert not np.any(split.E)
    assert not np.any(split.H)


@pytest.mark.parametrize(
    "problem", ("cpml", "alignment", "outside scatterer", "lumped")
)
def test_tfsf_rejects_invalid_layout_before_advancing(problem, media):
    grid = Grid((22, 22, 1), 1e-3)
    grid.add(CPMLBoundary(3))
    lower, upper = [0.005, 0.005, 0], [0.016, 0.016, 0]
    if problem == "cpml":
        lower[0] = 0.003
    elif problem == "alignment":
        lower[0] = 0.0055
    elif problem == "outside scatterer":
        grid.cell_material[4:6, 10, 0, 0] = 2
    else:
        grid.add(Resistor(0.001, 0.001, 0, 0.002, 0.001, 0, 50, direction=Direction.X))
    grid.add(
        TFSFSource(
            *lower, *upper, wave=PlaneWave(lambda t: 1, material=media["vacuum"])
        )
    )
    with pytest.raises(ValueError):
        grid.run(1, progress=False)
    assert grid.current_time_step == 0
    assert not np.any(grid.E)
    assert not np.any(grid.H)


@pytest.mark.parametrize("kind", (ScatteredFieldSource, TFSFSource))
def test_incident_wave_requires_compatible_grid_and_background(kind, media):
    def add_source(grid, wave):
        grid.add(
            kind(wave)
            if kind is ScatteredFieldSource
            else kind(0.005, 0.005, 0, 0.016, 0.016, 0, wave=wave)
        )

    for problem in (
        "inactive",
        "bloch",
        "periodic propagation",
        "cpml material",
        "cfl",
        "multiple",
    ):
        grid = Grid((22, 22, 1), 1e-3)
        wave = PlaneWave(lambda t: 1, material=media["vacuum"])
        if problem == "inactive":
            wave = PlaneWave(
                lambda t: 1,
                material=media["vacuum"],
                direction=(0, 0, 1),
                polarization=(1, 0, 0),
            )
        elif problem == "bloch":
            grid.add(PeriodicBlochBoundary((0, 20, 0), y_direction=True))
        elif problem == "periodic propagation":
            grid.add(PeriodicBoundary(x_direction=True))
        elif problem == "cpml material":
            grid.add(CPMLBoundary(3))
            grid.cell_material[1, 1, 0, 0] = 2
        elif problem == "cfl":
            wave = PlaneWave(lambda t: 1, material=media["fast"])
        else:
            add_source(grid, wave)
        add_source(grid, wave)
        with pytest.raises(ValueError):
            grid.run(1, progress=False)
        assert grid.current_time_step == 0


def test_transverse_periodic_tfsf_surface_and_scattered_lumped_rejection(media):
    grid = Grid((64, 6, 1), 1e-3)
    grid.add(PeriodicBoundary(y_direction=True))
    grid.add(CPMLBoundary(6, faces=("x-", "x+")))
    source = TFSFSource(
        0.01,
        0,
        0,
        0.05,
        0.005,
        0,
        wave=PlaneWave(GaussianWaveform(1e-10, 2e-11), material=media["vacuum"]),
    )
    grid.add(source)
    grid.run(90, progress=False)
    np.testing.assert_allclose(grid.E[:, 0, 0], grid.E[:, -1, 0], atol=1e-14)
    grid.reset()
    source.y_max = 0.004
    with pytest.raises(ValueError, match="span"):
        grid.run(1, progress=False)
    grid = Grid((6, 1, 1), 1e-3)
    grid.add(Resistor(0, 0, 0, 0.002, 0, 0, 50, direction=Direction.X))
    grid.add(ScatteredFieldSource(PlaneWave(lambda t: 1, material=media["vacuum"])))
    with pytest.raises(ValueError, match="lumped"):
        grid.run(1, progress=False)
