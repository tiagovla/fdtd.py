import argparse

import numpy as np

from fdtd import (
    CPMLBoundary,
    EFieldDetector,
    GaussianWaveform,
    Grid,
    ImpressedElectricCurrentSource,
)
from fdtd.constants import SPEED_LIGHT
from fdtd.plotting import pyplot


def simulate(cells, steps, pml, *, progress=True):
    grid = Grid((cells, cells, 1), spacing=1e-3)
    grid.add(CPMLBoundary(thickness=pml))
    interior = cells - 2 * pml
    source_x = (pml + interior // 3) * grid.dx
    detector_x = (pml + 2 * interior // 3) * grid.dx
    center_y = (cells // 2) * grid.dy
    tau = 8 * grid.dt
    waveform = GaussianWaveform(t_0=4.5 * tau, tau=tau)
    source = ImpressedElectricCurrentSource(
        source_x, center_y, 0, source_x, center_y, 0, waveform=waveform
    )
    detector = EFieldDetector(
        detector_x, center_y, 0, detector_x, center_y, 0, name="Ez probe"
    )
    grid.add(source)
    grid.add(detector)

    arrival = waveform.t_0 + (detector_x - source_x) / SPEED_LIGHT
    snapshot_step = min(steps, max(1, int(np.ceil(arrival / grid.dt))))
    grid.run(snapshot_step, progress=progress)
    snapshot = grid.E[:, :, 0, 2].real.copy()
    snapshot_time = grid.time_passed
    if steps > snapshot_step:
        grid.run(steps - snapshot_step, progress=progress)
    return grid, source, detector, snapshot, snapshot_time


def plot_results(grid, source, detector, snapshot, snapshot_time, pml):
    plt = pyplot()
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, width_ratios=(1.2, 1))
    field_ax = fig.add_subplot(layout[:, 0])
    source_ax = fig.add_subplot(layout[0, 1])
    detector_ax = fig.add_subplot(layout[1, 1], sharex=source_ax)

    limit = np.max(np.abs(snapshot)) or 1
    image = field_ax.imshow(
        snapshot.T,
        origin="lower",
        extent=(
            -grid.dx / 2 * 1e3,
            (grid.x[-1] + grid.dx / 2) * 1e3,
            -grid.dy / 2 * 1e3,
            (grid.y[-1] + grid.dy / 2) * 1e3,
        ),
        cmap="bwr",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    x_max, y_max = grid.x[-1] * 1e3, grid.y[-1] * 1e3
    for start, end in ((0, pml * grid.dx * 1e3), (x_max - pml * grid.dx * 1e3, x_max)):
        field_ax.axvspan(start, end, color="gray", alpha=0.25)
    for start, end in ((0, pml * grid.dy * 1e3), (y_max - pml * grid.dy * 1e3, y_max)):
        field_ax.axhspan(start, end, color="gray", alpha=0.25)
    field_ax.plot(source.x_min * 1e3, source.y_min * 1e3, "k*", label="Gaussian source")
    field_ax.plot(
        detector.x_min * 1e3,
        detector.y_min * 1e3,
        "ko",
        markerfacecolor="white",
        label="Detector",
    )
    field_ax.set(
        xlim=(0, x_max),
        ylim=(0, y_max),
        xlabel="x [mm]",
        ylabel="y [mm]",
        title=f"Ez at {snapshot_time * 1e9:.3f} ns (shaded: CPML)",
    )
    field_ax.legend(loc="upper right")
    fig.colorbar(image, ax=field_ax, label="Ez [V/m]", shrink=0.8)

    count = detector.sample_count
    times = detector.time[:count]
    source_times = times - grid.dt / 2
    source_ax.plot(source_times * 1e9, source.waveform(source_times), color="red")
    source_ax.set(title="Gaussian excitation", ylabel="Jz [A/m²]")
    detector_ax.plot(
        times * 1e9, detector.captured.reshape(-1, count)[0].real, color="blue"
    )
    detector_ax.set(
        title="Electric field at detector", xlabel="Time [ns]", ylabel="Ez [V/m]"
    )
    for ax in (source_ax, detector_ax):
        ax.axvline(
            snapshot_time * 1e9, color="gray", linestyle="--", label="Field snapshot"
        )
        ax.set_xlim(0, grid.time_passed * 1e9)
        ax.grid(alpha=0.3)
    source_ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="2D Gaussian pulse with CPML absorption"
    )
    parser.add_argument("--cells", type=int, default=96)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--pml", type=int, default=12)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.pml < 1:
        parser.error("--pml must be positive")
    if args.cells < 2 * args.pml + 9:
        parser.error("--cells must be at least 2 * --pml + 9")
    if args.steps < 1:
        parser.error("--steps must be positive")

    result = simulate(args.cells, args.steps, args.pml, progress=not args.no_plot)
    if not args.no_plot:
        fig = plot_results(*result, args.pml)
        pyplot().show()
        pyplot().close(fig)
    print(f"Completed {result[0].current_time_step} steps")


if __name__ == "__main__":
    main()
