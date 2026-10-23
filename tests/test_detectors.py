import numpy as np
import pytest

from fdtd import EFieldDetector, Grid, HFieldDetector, VoltageDetector
from fdtd.utils import Direction


def test_voltage_detector_preserves_complex_values():
    grid = Grid(shape=(3, 3, 3), spacing=1e-3)
    detector = VoltageDetector(0, 0, 0, 0, 0, 1e-3)
    grid.add(detector)
    grid.n_steps = 1
    detector.attach_to_grid()

    grid.E[:, :, :, 2] = 1 + 2j
    detector.update()

    assert detector.captured.dtype == complex
    assert detector.captured[0].imag != 0


@pytest.mark.parametrize("kind, offset", [(EFieldDetector, 1), (HFieldDetector, 0.5)])
def test_spectrum_preserves_spatial_channels_and_physical_phase(kind, offset):
    grid = Grid((4, 4, 4), 1e-3)
    detector = kind(0, 0, 0, 1e-3, 0, 0)
    grid.add(detector)
    grid.n_steps = 32
    detector.attach_to_grid()
    frequency = 4 / (grid.n_steps * grid.dt)
    wave = np.exp(2j * np.pi * frequency * detector.time)
    detector.captured[0, 0, 0] = wave
    detector.captured[1, 0, 0] = 2 * wave
    grid.current_time_step = 32
    freq, spectrum = detector.post_process()
    np.testing.assert_allclose(detector.time, (np.arange(32) + offset) * grid.dt)
    assert freq[4] == pytest.approx(frequency)
    np.testing.assert_allclose(spectrum[:, 0, 0, 4], [1, 2], atol=1e-14)
    _, averaged = detector.post_process(aggregation="mean")
    assert averaged[4] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "kind, direction, expected",
    [
        (EFieldDetector, Direction.X, 1.5),
        (HFieldDetector, Direction.Z, 3),
    ],
)
def test_interpolated_probe_uses_surrounding_staggered_components(
    kind, direction, expected
):
    grid = Grid((4, 4, 4), 1e-3)
    field = grid.E if kind is EFieldDetector else grid.H
    i, j, _ = np.indices(grid.shape)
    field[..., direction.value] = i if kind is EFieldDetector else i + j
    detector = kind(
        2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3, direction=direction, interpolate=True
    )
    grid.add(detector)
    grid.n_steps = 1
    detector.attach_to_grid()
    detector.update()
    assert detector.captured.item() == pytest.approx(expected)


def test_decimated_reduction_preserves_sampling_across_split_runs():
    grid = Grid((4, 4, 4), 1e-3)
    detector = EFieldDetector(
        0, 0, 0, 3e-3, 3e-3, 3e-3, sample_interval=3, reduction="mean"
    )
    grid.add(detector)
    grid.run(4, progress=False)
    grid.run(4, progress=False)
    assert detector.captured.shape == (3,)
    np.testing.assert_allclose(detector.time, np.array([1, 4, 7]) * grid.dt)
    assert detector.sample_count == 3
    detector.post_process()
    assert (
        detector.estimate_memory(8) == detector.captured.nbytes + detector.time.nbytes
    )


def test_large_capture_fails_before_allocating_storage():
    grid = Grid((20, 20, 1), 1e-3)
    detector = EFieldDetector(0, 0, 0, 19e-3, 19e-3, 0, max_memory_bytes=1024)
    grid.add(detector)
    assert grid.estimate_memory(100) > 1024
    with pytest.raises(MemoryError, match="sample_interval"):
        grid.run(100, progress=False)
    assert detector.captured is None
    assert grid.current_time_step == 0
