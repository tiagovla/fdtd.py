from abc import ABC, abstractproperty


class Source(ABC):
    """An source to be placed in the grid."""

    def __init__(self):
        """Initialize the source."""
        pass

    @abstractproperty
    def name(self) -> str:
        """Return the name of the source."""

    def __repr__(self):
        """Dev. string representation."""
        return f"{Source}({self.name})"

    def attach_to_grid(self):
        """Attach object to grid."""
        pass


class VoltageSource(Source):
    """Implement a voltage source."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
        waveform_type: str = "unit_step",
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = (resistance, )
        self.waveform_type = waveform_type


class CurrentSource(Source):
    """Implement a current source."""

    def __init__(
        self,
        x_min: float,
        y_min: float,
        z_min: float,
        x_max: float,
        y_max: float,
        z_max: float,
        resistance: float = 50,
        waveform_type: str = "unit_step",
    ):
        """Initialize the object."""
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.resistance = (resistance, )
        self.waveform_type = waveform_type
