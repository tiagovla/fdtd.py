import numpy as np
import pytest

from fdtd import Capacitor, Diode, Grid, Inductor, Resistor
from fdtd.constants import FREESPACE_PERMITTIVITY as EPS_0
from fdtd.utils import Direction


@pytest.mark.parametrize("element", [Capacitor, Inductor, Diode, Resistor])
def test_lumped_elements_initialize_base_state(element):
    device = element(0, 0, 0, 1e-3, 1e-3, 1e-3, name="device",
                     direction=Direction.X)

    assert device.name == "device"
    assert device.direction is Direction.X
    assert device.grid is None
    assert device.bounding_box.x_max == 1e-3


@pytest.mark.parametrize("element, value", [(Resistor, 50), (Capacitor, 1e-9)])
@pytest.mark.parametrize("direction", list(Direction))
def test_terminal_value_is_preserved_across_geometry(element, value, direction):
    grid = Grid((6, 6, 6), (1e-3, 1.5e-3, 2e-3))
    axis = direction.value
    end = np.array(grid.spacing) * 2
    end[axis] = grid.spacing[axis] * 3
    device = element(0, 0, 0, *end, value, direction=direction)
    grid.add(device)
    grid._prepare()
    length = grid.spacing[axis]
    area = np.prod([grid.spacing[d] for d in range(3) if d != axis])
    effective = value * 9 / 3 if element is Resistor else value * 3 / 9
    term = grid.dt * length / (effective * area) if element is Resistor else 2 * effective * length / area
    expected = 2 * grid.dt / (2 * EPS_0 + term)
    index = (0, 0, 0, axis)
    assert grid.c_eh[index] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("direction", list(Direction))
def test_inductor_current_recurrence_and_reset(direction):
    grid = Grid((4, 4, 4), 1e-3)
    end = [0, 0, 0]
    end[direction.value] = 2e-3
    device = Inductor(0, 0, 0, *end, inductance=1e-8, direction=direction)
    grid.add(device)
    grid._prepare()
    grid.E[device.index] = 2
    device.update_E()
    expected_current = grid.dt * 1e-3 / ((1e-8 / 2) * 1e-6) * 2
    np.testing.assert_allclose(device.current_density, expected_current)
    device.update_E()
    assert np.all(grid.E[device.index].real < 2)
    grid.reset()
    assert device.current_density is None


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("drive", [-1.0, 1.0, 1000.0])
def test_diode_satisfies_implicit_junction_equation(direction, polarity, drive):
    grid = Grid((3, 3, 3), 1e-3)
    end = [0, 0, 0]
    end[direction.value] = 1e-3
    diode = Diode(0, 0, 0, *end, direction=direction, polarity=polarity)
    grid.add(diode)
    grid._prepare()
    grid.E[diode.index] = polarity * drive / 1e-3
    diode.update_E()
    voltage = diode.previous_voltage
    current = diode.saturation_current * np.expm1(voltage / (2 * diode._thermal_voltage))
    assert voltage - drive + diode._gamma * current == pytest.approx(0, abs=1e-8)
    if drive < 0:
        assert voltage == pytest.approx(drive)
    else:
        assert 0 < voltage < drive
    grid.reset()
    assert diode.previous_voltage == 0


def test_diode_rejects_complex_fields_and_reports_failed_solve():
    grid = Grid((3, 3, 3), 1e-3)
    diode = Diode(0, 0, 0, 0, 0, 1e-3, max_iterations=1)
    grid.add(diode)
    grid._prepare()
    grid.E[diode.index] = 1j
    with pytest.raises(ValueError, match="real"):
        diode.update_E()
    grid.E[diode.index] = 1000
    with pytest.raises(RuntimeError, match="converge"):
        diode.update_E()
