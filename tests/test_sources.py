import numpy as np
import pytest

from fdtd import Grid, VoltageSource
from fdtd.constants import FREESPACE_PERMITTIVITY as EPS_0
from fdtd.utils import Direction


@pytest.mark.parametrize(
    "direction, end",
    [
        (Direction.X, (1e-3, 0, 0)),
        (Direction.Y, (0, 1e-3, 0)),
        (Direction.Z, (0, 0, 1e-3)),
    ],
)
def test_voltage_source_supports_each_direction(direction, end):
    grid = Grid(shape=(3, 3, 3), spacing=1e-3)
    source = VoltageSource(
        0, 0, 0, *end, waveform=lambda time: 1.0, direction=direction
    )
    grid.add(source)
    grid._prepare()

    source.update_E()

    assert source._v_f == 1
    field = grid.E[source.i_s, source.j_s, source.k_s, direction.value]
    expected = -(2 * grid.dt) / (2 * EPS_0 + grid.dt / (50 * 1e-3)) / (50 * 1e-6)
    np.testing.assert_allclose(field, expected, rtol=1e-12)
    assert np.all(field.real < 0)


def test_voltage_drive_is_sampled_at_half_step():
    times = []
    grid = Grid((3, 3, 3), 1e-3)
    grid.add(
        VoltageSource(
            0, 0, 0, 0, 0, 1e-3, waveform=lambda time: times.append(time) or 1
        )
    )
    grid.run(3, progress=False)
    np.testing.assert_allclose(times, (np.arange(3) + 0.5) * grid.dt)
