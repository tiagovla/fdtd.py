import numpy as np
from _bandgap import arguments, dielectric, draw, path

from fdtd import (
    EFieldDetector,
    GaussianWaveform,
    Grid,
    HFieldDetector,
    ImpressedElectricCurrentSource,
    ImpressedMagneticCurrentSource,
    PeriodicBlochBoundary,
    Sphere,
)


def plane_wave_bands(bz_path, period, radius, eps_rod, grid_size, order):
    from morpho import Geometry, Solver

    geometry = Geometry((period, 0), (0, period), grid_size, grid_size)

    @geometry.overwrite
    def eps_rf(eps_r, x, y):
        eps_r[x**2 + y**2 <= radius**2] = eps_rod

    tm_solver = Solver(geometry, bz_path, P=order, Q=order, pol="TM")
    te_solver = Solver(geometry, bz_path, P=order, Q=order, pol="TE")
    tm_solver.run()
    te_solver.run()
    scale = period / (2 * np.pi)
    return [
        (bz_path.betas.cumsum, np.vstack(tm_solver.wn) * scale, "TM"),
        (bz_path.betas.cumsum, np.vstack(te_solver.wn) * scale, "TE"),
    ]


def simulation(beta, cells, steps, probes, seed, magnetic=False):
    from scipy.signal import find_peaks

    period = 1e-6
    spacing = period / cells
    grid = Grid((cells, cells, 1), spacing)
    grid.add(
        Sphere(
            period / 2,
            period / 2,
            spacing / 2,
            0.2 * period,
            dielectric("bandgap-2d", 8.9),
        )
    )
    grid.add(PeriodicBlochBoundary((*beta, 0), x_direction=True, y_direction=True))
    rng = np.random.default_rng(seed)
    waveform = GaussianWaveform(45 * grid.dt, 10 * grid.dt)
    source_kind = (
        ImpressedMagneticCurrentSource if magnetic else ImpressedElectricCurrentSource
    )
    detector_kind = HFieldDetector if magnetic else EFieldDetector
    for _ in range(probes):
        point = (*((rng.integers(1, cells - 1, 2)) * spacing), 0)
        grid.add(source_kind(*point, *point, waveform=waveform))
        point = (*((rng.integers(1, cells - 1, 2)) * spacing), 0)
        grid.add(detector_kind(*point, *point))
    grid.run(steps, progress=False)
    spectra = [detector.post_process()[1] for detector in grid.detectors]
    power = np.sum(np.abs(spectra) ** 2, axis=0)
    peaks, _ = find_peaks(power)
    return grid.detectors[0].freq[peaks]


def main():
    args = arguments(2)
    bz_path = path(2, args.points, 1e-6)
    pwe = plane_wave_bands(
        bz_path, 1e-6, 0.2e-6, 8.9, args.morpho_grid, args.morpho_order
    )
    betas = bz_path.betas
    bands = []
    for magnetic, name in ((False, "TM"), (True, "TE")):
        for beta, distance in zip(betas.values.T, betas.cumsum):
            frequencies = simulation(
                beta, args.cells, args.steps, args.probes, args.seed, magnetic
            )
            bands.append((distance, frequencies, f"{name} FDTD"))
    draw(
        bz_path,
        bands,
        1e-6,
        not args.no_plot,
        y_limits=(0, 1),
        curves=pwe,
    )
    print(
        f"Completed {len(bands)} FDTD runs and "
        f"{sum(band[1].shape[1] for band in pwe)} PWE bands"
    )


if __name__ == "__main__":
    main()
