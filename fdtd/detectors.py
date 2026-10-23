"""Field and voltage sampling on a staggered grid."""
from abc import abstractmethod
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from .bases import FDTDElementBase
from .utils import BoundingBox, Direction


class Detector(FDTDElementBase):
    """Record a component over a bounding box, optionally interpolated to nodes."""

    _field_name = "E"
    _voltage = False

    def __init__(self, x_min, y_min, z_min, x_max, y_max, z_max,
                 name=None, direction=Direction.Z, plot=False, *, interpolate=False):
        super().__init__()
        self.bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max
        self.name = name or self._create_new_name()
        if not isinstance(direction, Direction):
            raise ValueError("direction must be a Direction value")
        self.direction = direction
        self.interpolate = interpolate
        self._plot = plot
        self.captured = None

    def reset(self):
        self.captured = None
        for name in ("time", "freq", "values_time", "values_freq"):
            self.__dict__.pop(name, None)

    @abstractmethod
    def _sample(self):
        """Return the next scalar or spatial array of samples."""

    def attach_to_grid(self):
        if self.grid is None:
            raise RuntimeError("Register the detector with a grid before preparing it.")
        coordinates = (self.grid.x, self.grid.y, self.grid.z)
        self.idx_s = tuple(int(np.argmin(abs(a - b))) for a, b in zip(
            coordinates, (self.x_min, self.y_min, self.z_min)))
        self.idx_e = tuple(int(np.argmin(abs(a - b))) for a, b in zip(
            coordinates, (self.x_max, self.y_max, self.z_max)))
        axis = self.direction.value
        self.slices = tuple(slice(s, e + (0 if self._voltage and d == axis else 1))
                            for d, (s, e) in enumerate(zip(self.idx_s, self.idx_e)))
        self.i_s, self.j_s, self.k_s = self.slices
        self.size = tuple(s.stop - s.start for s in self.slices)
        if any(n <= 0 for n in self.size):
            raise ValueError("Voltage detector must span at least one field edge.")
        self.c_v = -self.grid.spacing[axis] / np.prod(
            [self.size[d] for d in range(3) if d != axis])
        shape = (() if self._voltage else self.size) + (self.grid.n_steps,)
        previous = self.captured
        self.captured = np.zeros(shape, dtype=getattr(self.grid, self._field_name).dtype)
        if previous is not None:
            count = min(previous.shape[-1], shape[-1])
            self.captured[..., :count] = previous[..., :count]
        offset = 0.5 if self._field_name == "H" else 1.0
        self.time = (np.arange(self.grid.n_steps) + offset) * self.grid.dt

    def _field_values(self):
        field = getattr(self.grid, self._field_name)[..., self.direction.value]
        if not self.interpolate:
            return field[self.slices]
        axes = ([self.direction.value] if self._field_name == "E" else
                [d for d in range(3) if d != self.direction.value])
        values = np.zeros(self.size, dtype=field.dtype)
        for shifts in product((0, 1), repeat=len(axes)):
            indices = [np.arange(s.start, s.stop) for s in self.slices]
            for axis, shift in zip(axes, shifts):
                indices[axis] = np.maximum(indices[axis] - shift, 0)
            values += field[np.ix_(*indices)] / (2 ** len(axes))
        return values

    def update(self):
        if self.captured is None:
            raise RuntimeError("Prepare detector storage with Grid.run() before sampling.")
        step = self.grid.current_time_step
        if step >= self.captured.shape[-1]:
            raise RuntimeError("Detector storage exhausted; advance with Grid.run().")
        self.captured[..., step] = self._sample()

    def post_process(self, aggregation=None, normalization="amplitude", positive_only=True):
        """Transform the time axis; return amplitudes or a time-integral spectrum."""
        if self.captured is None or not self.captured.shape[-1]:
            raise RuntimeError("No detector samples are available.")
        count = min(self.grid.current_time_step, self.captured.shape[-1])
        if not count:
            raise RuntimeError("No detector samples are available.")
        values = self.captured[..., :count]
        if aggregation not in (None, "mean", "sum"):
            raise ValueError("aggregation must be None, 'mean', or 'sum'")
        if values.ndim > 1:
            axes = tuple(range(values.ndim - 1))
            if aggregation == "mean":
                values = values.mean(axis=axes)
            elif aggregation == "sum":
                values = values.sum(axis=axes)
            elif np.prod(values.shape[:-1]) == 1:
                values = values.reshape(count)
        self.values_time = values
        self.freq = np.fft.fftfreq(count, self.grid.dt)
        spectrum = np.fft.fft(values, axis=-1)
        if normalization == "amplitude":
            spectrum /= count
        elif normalization == "integral":
            spectrum *= self.grid.dt
        elif normalization != "none":
            raise ValueError("normalization must be 'amplitude', 'integral', or 'none'")
        spectrum *= np.exp(-2j * np.pi * self.freq * self.time[0])
        if positive_only:
            indices = self.freq >= 0
            self.freq = self.freq[indices]
            spectrum = spectrum[..., indices]
        self.values_freq = spectrum
        return self.freq, self.values_freq

    def pos_processing(self, **kwargs):
        """Compatibility spelling for post_process()."""
        return self.post_process(**kwargs)

    def plot(self, component="real"):
        if not self._plot:
            return
        if self.captured is None:
            raise RuntimeError("No detector samples are available.")
        transforms = {"real": np.real, "imag": np.imag,
                      "magnitude": np.abs, "phase": np.angle}
        if component not in transforms:
            raise ValueError("component must be real, imag, magnitude, or phase")
        count = min(self.grid.current_time_step, self.captured.shape[-1])
        values = transforms[component](self.captured[..., :count])
        _, ax = plt.subplots()
        ax.plot(self.time[:count], values.reshape(-1, count).T)
        ax.set_xlabel("Time [s]")
        plt.show()

    def plot_3d(self, ax, alpha=0.5):
        x, y, z = np.meshgrid([self.x_min, self.x_max],
                              [self.y_min, self.y_max], [self.z_min, self.z_max])
        for dim in range(3):
            for edge in (0, -1):
                index = [slice(None)] * 3
                index[dim] = edge
                index = tuple(index)
                ax.plot_surface(x[index], y[index], z[index], alpha=alpha, color="#B0BF00")


class VoltageDetector(Detector):
    """Negative electric line integral averaged over parallel paths."""

    _voltage = True

    def _sample(self):
        return self.c_v * np.sum(self.grid.E[self.slices + (self.direction.value,)])


class EFieldDetector(Detector):
    def _sample(self):
        return self._field_values()


class HFieldDetector(EFieldDetector):
    _field_name = "H"
