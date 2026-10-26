"""Voltage source and resistor connected through two conducting plates.

SIDE VIEW, looking along x (the plates extend from x=4 mm to x=6 mm):

    z [mm]
      ^
    8 |     +------------------+------------------+  upper plate
      |     |                  :                  |
      |  [Vs + Rs]         V(t) probe             [R]
      |     |                  :                  |
    4 |     +------------------+------------------+  lower plate
      |
      +-----|------------------|------------------|----> y [mm]
            2                  5                  8
          source             detector            load

    Vs = 1 V step at 10% of the run duration; Rs = 50 ohm; R = 50 ohm.
    The dotted probe is a measurement, not an electrical connection.

WHAT IS DEFINED BELOW

1. Grid: a 10 mm cube, with 1 mm cells and vacuum as the default background.
   No absorbing boundary is added, so the finite grid affects ringing.
2. Plates: two zero-thickness Brick objects, 2 mm wide and 6 mm long,
   separated by 4 mm. Their high conductivity approximates PEC sheets.
3. Connections: source, load, and probe span the gap in the z direction.
   The source is at the front end; the load is at the back end.
   VoltageSource includes Rs internally; Resistor represents the load R.
4. Measurement: VoltageDetector records -integral(Ez dz), averaged over
   parallel paths across the plate width, halfway between source and load.
5. Reference: the ideal DC divider is Vs*R/(Rs+R) = 0.5 V. Propagation,
   fringing fields, and inductance produce a different 3D transient, so the
   reference is a DC limit, not an exact charging curve.

All object coordinates are in meters. Bounding boxes use the order
(x_min, y_min, z_min, x_max, y_max, z_max).

Run:  uv run --extra plot examples/voltage_resistor.py
"""

import argparse
from dataclasses import dataclass

import numpy as np

from fdtd import (
    Brick,
    Grid,
    Material,
    Resistor,
    StepWaveform,
    VoltageDetector,
    VoltageSource,
)
from fdtd.plotting import pyplot

MM = 1e-3
GRID_SHAPE = (10, 10, 10)
SPACING = 1 * MM
PLATE = Material("voltage-resistor-PEC", sigma_e=1e10, color="#0000BC")
SOURCE_RESISTANCE = 50.0
SOURCE_VOLTAGE = 1.0
RESISTANCE = 50.0


@dataclass
class Comparison:
    """Keep the measured trace and DC reference together for plotting."""

    grid: Grid
    detector: VoltageDetector
    analytical: np.ndarray
    title: str
    reference_label: str = "Analytical DC divider (not transient)"
    dc_only: bool = True

    @property
    def error(self):
        """Distance to DC, including the physical startup transient."""
        return self.detector.captured.real - self.analytical


def build_grid(courant_factor, total_steps):
    """Build the complete layout before advancing any fields."""
    grid = Grid(GRID_SHAPE, spacing=SPACING, courant_factor=courant_factor)
    upper_plate = Brick(
        *(4 * MM, 2 * MM, 8 * MM),
        *(6 * MM, 8 * MM, 8 * MM),
        material=PLATE,
        name="Upper plate",
    )
    lower_plate = Brick(
        *(4 * MM, 2 * MM, 4 * MM),
        *(6 * MM, 8 * MM, 4 * MM),
        material=PLATE,
        name="Lower plate",
    )
    source = VoltageSource(
        *(4 * MM, 2 * MM, 4 * MM),
        *(6 * MM, 2 * MM, 8 * MM),
        waveform=StepWaveform(
            t_0=0.1 * total_steps * grid.dt,
            amplitude=SOURCE_VOLTAGE,
        ),
        resistance=SOURCE_RESISTANCE,
        name="Voltage source",
    )
    load = Resistor(
        *(4 * MM, 8 * MM, 4 * MM),
        *(6 * MM, 8 * MM, 8 * MM),
        resistance=RESISTANCE,
        name="Load resistor",
    )

    detector = VoltageDetector(
        *(4 * MM, 5 * MM, 4 * MM),
        *(6 * MM, 5 * MM, 8 * MM),
        name="Midpoint plate voltage",
    )
    for component in (upper_plate, lower_plate, source, load, detector):
        grid.add(component)
    return grid, detector


def simulate(steps=5000, *, courant_factor=0.9, progress=False):
    """Run the 3D model and evaluate its ideal steady-state circuit limit."""
    grid, detector = build_grid(courant_factor, steps)
    grid.run(steps, progress=progress)
    steady = SOURCE_VOLTAGE * RESISTANCE / (SOURCE_RESISTANCE + RESISTANCE)
    analytical = np.full(detector.time.shape, steady)
    return Comparison(
        grid,
        detector,
        analytical,
        "Voltage source + resistor between conducting plates",
    )


def plot_geometry(result):
    """Show the two plates and their source, load, and midpoint probe."""
    ax = result.grid.plot_geometry(show=False)
    ax.set_title("Voltage source and resistor between plates", fontsize=14)
    return ax


def plot_comparison(result):
    """Show measured voltage above its distance from the ideal DC limit."""
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
        ylabel="V − Vdc [V]",
    )
    for ax in (voltage, error):
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--courant-factor", type=float, default=0.9)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not np.isfinite(args.courant_factor) or not 0 < args.courant_factor < 1:
        parser.error("--courant-factor must be between 0 and 1")
    result = simulate(
        args.steps,
        courant_factor=args.courant_factor,
        progress=True,
    )
    fig = plot_comparison(result)
    geometry = plot_geometry(result)
    plt = pyplot()
    plt.show()
    plt.close(geometry.figure)
    plt.close(fig)
    error = result.error
    print(
        f"Completed {result.grid.current_time_step} steps; "
        f"dt = {result.grid.dt:.6g} s; "
        f"distance to analytical DC at final sample = {abs(error[-1]):.6g} V"
    )


if __name__ == "__main__":
    main()
