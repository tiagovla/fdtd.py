"""A 1 V step through Rs=50 ohm drives a Shockley diode with Rd=10 ohm.

Compare the settled terminal voltage with the analytical DC circuit equation:
Vs = Vj + (Rs+Rd)*Is*expm1(Vj/(n*VT)); Vterminal = Vs - Rs*Id.
The reference is a DC asymptote, not an analytical charging transient.
"""

import argparse
from dataclasses import dataclass

import numpy as np

from fdtd import Diode, Grid, StepWaveform, VoltageDetector, VoltageSource
from fdtd.constants import FREESPACE_PERMITTIVITY as EPS_0
from fdtd.plotting import pyplot
from fdtd.utils import Direction

BOUNDS = (0, 0, 0, 1e-3, 0, 0)
SPACING = (1e-3, 1e-2, 1e-2)
SOURCE_RESISTANCE = 50.0
SOURCE_VOLTAGE = 1.0
GRID_CAPACITANCE = EPS_0 * SPACING[1] * SPACING[2] / SPACING[0]

SERIES_RESISTANCE = 10.0
SATURATION_CURRENT = 1e-14
TEMPERATURE = 300.0
IDEALITY = 1.0


@dataclass
class Comparison:
    grid: Grid
    detector: VoltageDetector
    analytical: np.ndarray
    title: str
    reference_label: str = "Analytical transient"
    dc_only: bool = False

    @property
    def error(self):
        return self.detector.captured.real - self.analytical


def terminal(courant_factor):
    """Isolate a longitudinal Yee edge: curl E = curl H = 0.

    Its displacement current is a physical shunt capacitance epsilon*A/length.
    Source and loads share that edge, so this verifies lumped coupling without
    propagation delays, radiation, or wire inductance from a 3D layout.
    """
    grid = Grid((2, 1, 1), SPACING, courant_factor=courant_factor)
    grid.add(
        VoltageSource(
            *BOUNDS,
            waveform=StepWaveform(amplitude=SOURCE_VOLTAGE),
            resistance=SOURCE_RESISTANCE,
            direction=Direction.X,
        )
    )
    detector = VoltageDetector(*BOUNDS, name="Terminal voltage", direction=Direction.X)
    grid.add(detector)
    return grid, detector


def dc_voltage():
    """Evaluate the static Shockley/load-line intersection by bisection.

    This is independent of the FDTD diode's implicit time-step solver.
    The equivalent closed form uses the principal Lambert W branch.
    """
    thermal_voltage = IDEALITY * 1.380649e-23 * TEMPERATURE / 1.602176634e-19
    lower, upper = 0.0, SOURCE_VOLTAGE
    for _ in range(80):
        junction = (lower + upper) / 2
        current = SATURATION_CURRENT * np.expm1(junction / thermal_voltage)
        supplied = junction + (SOURCE_RESISTANCE + SERIES_RESISTANCE) * current
        if supplied > SOURCE_VOLTAGE:
            upper = junction
        else:
            lower = junction
    current = SATURATION_CURRENT * np.expm1(((lower + upper) / 2) / thermal_voltage)
    return SOURCE_VOLTAGE - SOURCE_RESISTANCE * current


def simulate(steps=2000, *, courant_factor=0.1, progress=False):
    grid, detector = terminal(courant_factor)
    grid.add(
        Diode(
            *BOUNDS,
            resistance=SERIES_RESISTANCE,
            saturation_current=SATURATION_CURRENT,
            temperature=TEMPERATURE,
            ideality=IDEALITY,
            direction=Direction.X,
            polarity=-1,  # VoltageDetector measures -integral(E dl).
        )
    )
    grid.run(steps, progress=progress)
    analytical = np.full(detector.time.shape, dc_voltage())
    return Comparison(
        grid,
        detector,
        analytical,
        "Voltage source + diode · Rs = 50 Ω, Rd = 10 Ω",
        reference_label="Analytical DC limit (not transient)",
        dc_only=True,
    )


def plot_comparison(result):
    """Overlay source, FDTD, and reference; distinguish DC distance from error."""
    plt = pyplot()
    fig, (voltage, error) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9, 6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    detector = result.detector
    detector.plot(ax=voltage, show=False, label="FDTD")
    time = detector.time[: detector.sample_count]
    voltage.plot(time, result.analytical, "--", label=result.reference_label)
    voltage.axhline(SOURCE_VOLTAGE, color="0.55", linestyle=":", label="Source step")
    voltage.set(title=result.title, xlabel="")
    voltage.legend(loc="best")
    error.plot(time, result.error, color="tab:red")
    error.axhline(0, color="0.55", linewidth=0.8)
    error.set(
        xlabel="Time [s]",
        ylabel="V − Vdc [V]" if result.dc_only else "Error [V]",
    )
    for ax in (voltage, error):
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--courant-factor", type=float, default=0.1)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not np.isfinite(args.courant_factor) or not 0 < args.courant_factor < 1:
        parser.error("--courant-factor must be between 0 and 1")
    result = simulate(
        args.steps,
        courant_factor=args.courant_factor,
        progress=not args.no_plot,
    )
    if not args.no_plot:
        fig = plot_comparison(result)
        pyplot().show()
        pyplot().close(fig)
    error = result.error
    metric = (
        f"distance to analytical DC at final sample = {abs(error[-1]):.6g} V"
        if result.dc_only
        else f"max error = {np.max(abs(error)):.6g} V; RMS error = {np.sqrt(np.mean(error**2)):.6g} V"
    )
    print(
        f"Completed {result.grid.current_time_step} steps; "
        f"dt = {result.grid.dt:.6g} s; Cgrid = {GRID_CAPACITANCE * 1e12:.6g} pF; {metric}"
    )


if __name__ == "__main__":
    main()
