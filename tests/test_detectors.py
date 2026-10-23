import numpy as np

from fdtd import Grid, VoltageDetector


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
