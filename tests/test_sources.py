import pytest

from fdtd import Grid, SineWaveform, VoltageSource
from fdtd.utils import Direction


@pytest.mark.parametrize("direction, end", [
    (Direction.X, (1e-3, 0, 0)),
    (Direction.Y, (0, 1e-3, 0)),
    (Direction.Z, (0, 0, 1e-3)),
])
def test_voltage_source_supports_each_direction(direction, end):
    grid = Grid(shape=(3, 3, 3), spacing=1e-3)
    source = VoltageSource(0, 0, 0, *end,
                           waveform=SineWaveform(frequency=1e9),
                           direction=direction)
    grid.add(source)
    grid._prepare()

    source.update_E()

    assert source._v_f == 1
