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
            """Wrap function."""
            return 0 * time

        return wrap

    def __call__(self, time: Union[float,
                                   np.ndarray]) -> Union[float, np.ndarray]:
        """Call a waveform function."""
        return self.func()(time)

    def plot(self, time: np.ndarray):
        """Plot waveform for a given time array."""
        y = self.func()(time)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, y)
        plt.show()


class GaussianWaveform(Waveform):
    """
    Model of a gaussian waveform.

    Represent a function ``A*exp(-(t-t0)^2/tau^2)``.
    This model should be attached to a source.

    Parameters
    ----------
    t_0 : float
        Initial time.
    tau : float
        Tau parameter.
    amplitude : float
        Amplitude.
    """

    def __init__(self, t_0: float, tau: float, amplitude: float = 1):
        """Initialize waveform."""
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
    """
    Model of a sine waveform.

    Represent a function ``A*sin(2*pi*freq*t-offset)``
    This model should be attached to a source.

    Parameters
    ----------
    frequency : float
        Frequency [Hz].
    offset : float
        Phase offset.
    amplitude : float
        Amplitude.
    """

    def __init__(self,
                 frequency: float,
                 offset: float = 0,
                 amplitude: float = 1):
        """Initialize waveform."""
        super().__init__()
        self.freq = frequency
        self.offset = offset
        self.amp = amplitude

    def func(self):
        """Return a gaussian waveform function."""

        def wrap(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            """Wrap function."""
            return self.amp * \
                    np.sin(2 * np.pi * self.freq * time - self.offset)

        return wrap


class StepWaveform(Waveform):
    """
    Model of a step waveform.

    Represent a function ``A*step(t-t0)``.
    This model should be attached to a source.

    Parameters
    ----------
    t_0 : float
        Initial time.
    amplitude : float
        Amplitude.
    """

    def __init__(self, t_0: float = 0, amplitude: float = 1):
        """Initialize waveform."""
        super().__init__()
        self.t_0 = t_0
        self.amp = amplitude

    def func(self):
        """Return a step waveform function."""

        def wrap(time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            return self.amp * np.heaviside(time - self.t_0, 1)

        return wrap
