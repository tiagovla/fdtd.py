from abc import ABC, abstractproperty


class LumpedElement(ABC):
    """A lumped element to be placed in the grid."""

    def __init__(self):
        """Initialize the lumped element."""
        pass

    @abstractproperty
    def name(self) -> str:
        """Return the name of the lumped element."""

    def __repr__(self):
        """Dev. string representation."""
        return f"{LumpedElement}({self.name})"

    def attach_to_grid(self):
        """Attach object to grid."""
        pass


class Resistor(LumpedElement):
    """Implement a resistor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


class Capacitor(LumpedElement):
    """Implement a capacitor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


class Inductor(LumpedElement):
    """Implement a inductor element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance


class Diode(LumpedElement):
    """Implement a diode element."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = resistance
