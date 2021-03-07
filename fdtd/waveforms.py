"""This module implements waveforms to be attached to sources."""
from abc import ABC, abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class Waveform(ABC):
    """Base Waveform."""

    @abstractmethod
    def func(self):
        """Waveform function."""

        def wrap(time):
            return 0 * time

        return wrap

    def __call__(self, time: Union[float,
                                   np.ndarray]) -> Union[float, np.ndarray]:
        """Implement call method."""
        return self.func()(time)

    def plot(self, time: np.ndarray):
        """Plot waveform for a given time array."""
        y = self.func()(time)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, y)
        plt.show()


class GaussianWaveform(Waveform):
    """Represent a gaussian waveform to be attached to a source."""

    def __init__(self, tau: float, t_0: float, amplitude: float = 1):
        """Represent a A*e^(-(t-t_0)/tau^2) function."""
        super().__init__()
        self.tau = tau
        self.t_0 = t_0
        self.amp = amplitude

    def func(self):
        """Return a gaussian waveform function."""

        def wrap(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return self.amp * np.exp(-(((time - self.t_0) / self.tau)**2))

        return wrap


class SineWaveform(Waveform):
    """Represent a sine waveform to be attached to a source."""

    def __init__(self, freq: float, offset: float, amplitude: float = 1):
        """Represent a A*sin(2*pi*f*t+offset)function."""
        super().__init__()
        self.freq = freq
        self.offset = offset
        self.amp = amplitude

    def func(self):
        """Return a gaussian waveform function."""

        def wrap(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return self.amp * np.sin(2 * np.pi * self.freq * time +
                                     self.offset)

        return wrap


class StepWaveform(Waveform):
    """Represent a sine waveform to be attached to a source."""

    def __init__(self, t_0: float, amplitude: float = 1):
        """Represent a A*sin(2*pi*f*t+offset)function."""
        super().__init__()
        self.t_0 = t_0
        self.amp = amplitude

    def func(self):
        """Return a gaussian waveform function."""

        def wrap(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return np.heaviside(time - self.t_0, 1)

        return wrap
