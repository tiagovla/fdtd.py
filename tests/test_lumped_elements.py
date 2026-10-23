import pytest

from fdtd import Capacitor, Diode, Inductor, Resistor
from fdtd.utils import Direction


@pytest.mark.parametrize("element", [Capacitor, Inductor, Diode, Resistor])
def test_lumped_elements_initialize_base_state(element):
    device = element(0, 0, 0, 1e-3, 1e-3, 1e-3, name="device",
                     direction=Direction.X)

    assert device.name == "device"
    assert device.direction is Direction.X
    assert device.grid is None
    assert device.bounding_box.x_max == 1e-3
