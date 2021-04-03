import numpy as np
import pytest

from fdtd.waveforms import GaussianWaveform, SineWaveform, StepWaveform


def test_sinewaveform():
    """
    Test a SineWaveform object.

    Ensure that the object is callable with the correct sine waveform.
    """
    t = np.random.rand(1000)
    freq, offset, amp = 1e9, 1, 2
    wf = SineWaveform(frequency=freq, offset=offset, amplitude=amp)
    y = amp * np.sin(2 * np.pi * freq * t - offset)

    assert y == pytest.approx(wf(t), rel=0.01)


def test_stepwaveform():
    """
    Test a StepWaveform object.

    Ensure that the object is callable with the correct step waveform.
    """
    t = np.random.uniform(0, 5, 1000)
    t_0, amp = 3, 2
    wf = StepWaveform(t_0=t_0, amplitude=amp)
    y = amp * np.heaviside(t - t_0, 1)
    assert y == pytest.approx(wf(t), rel=0.01)


def test_gaussianwaveform():
    """
    Test a GaussianWaveform object.

    Ensure that the object is callable with the correct gaussian waveform.
    """
    t = np.random.uniform(0, 5, 1000)
    t_0, tau, amp = 3, 4, 2
    wf = GaussianWaveform(t_0=t_0, tau=tau, amplitude=amp)
    y = amp * np.exp(-(((t-t_0) / tau)**2))
    assert y == pytest.approx(wf(t), rel=0.01)
