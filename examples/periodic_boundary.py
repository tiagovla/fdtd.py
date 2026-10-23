import argparse

from fdtd import (
    EFieldDetector,
    GaussianWaveform,
    Grid,
    ImpressedElectricCurrentSource,
    PeriodicBoundary,
)


def main():
    parser = argparse.ArgumentParser(description="Periodic two-dimensional propagation")
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.cells < 4:
        parser.error("--cells must be at least 4")
    grid = Grid((args.cells, args.cells, 1), 1e-3)
    tau = 10 * grid.dt
    source = args.cells // 4 * 1e-3
    probe = args.cells * 3 // 4 * 1e-3
    grid.add(
        ImpressedElectricCurrentSource(
            source,
            source,
            0,
            source,
            source,
            0,
            waveform=GaussianWaveform(4.5 * tau, tau),
        )
    )
    grid.add(EFieldDetector(probe, probe, 0, probe, probe, 0, plot=not args.no_plot))
    grid.add(PeriodicBoundary(x_direction=True, y_direction=True))
    grid.run(args.steps, progress=not args.no_plot)
    print(f"Completed {grid.current_time_step} steps")


if __name__ == "__main__":
    main()
