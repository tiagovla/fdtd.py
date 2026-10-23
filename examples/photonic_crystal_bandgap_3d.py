import concurrent.futures
import multiprocessing

import numpy as np
from _bandgap import arguments, dielectric, draw, path

from fdtd import (
    Brick,
    EFieldDetector,
    GaussianWaveform,
    Grid,
    ImpressedElectricCurrentSource,
    PeriodicBlochBoundary,
)
from fdtd.utils import Direction


def plane_wave_bands(bz_path, period, width, eps_rod, grid_size, order):
    from morpho import Geometry, Solver

    vectors = ((period, 0, 0), (0, period, 0), (0, 0, period))
    geometry = Geometry(*vectors, grid_size, grid_size, grid_size)

    @geometry.overwrite
    def eps_rf(eps_r, x, y, z):
        positions = np.mod((x, y, z), period)
        for long_axis in range(3):
            transverse = [axis for axis in range(3) if axis != long_axis]
            for first in (0, period - width):
                for second in (0, period - width):
                    mask = (
                        (positions[transverse[0]] >= first)
                        & (positions[transverse[0]] < first + width)
                        & (positions[transverse[1]] >= second)
                        & (positions[transverse[1]] < second + width)
                    )
                    eps_r[mask] = eps_rod

    solver = Solver(geometry, bz_path, P=order, Q=order, R=order)
    solver.run()
    frequencies = np.vstack(solver.wn) * period / (2 * np.pi)
    return [(bz_path.betas.cumsum, frequencies, "3D Morpho.py")]


def simulation(job):
    from scipy.signal import find_peaks

    beta, cells, steps, probes, seed = job
    period = 1e-6
    width = 0.1 * period
    spacing = period / cells
    grid = Grid((cells, cells, cells), spacing, courant_factor=0.5)
    material = dielectric("bandgap-3d", 2.43)
    for long_axis in range(3):
        transverse = [d for d in range(3) if d != long_axis]
        for first in (0, period - width):
            for second in (0, period - width):
                lower, upper = np.zeros(3), np.full(3, period)
                lower[transverse] = (first, second)
                upper[transverse] = (first + width, second + width)
                grid.add(Brick(*lower, *upper, material))
    grid.add(PeriodicBlochBoundary(tuple(beta), True, True, True))
    rng = np.random.default_rng(seed)
    waveform = GaussianWaveform(45 * grid.dt, 10 * grid.dt)
    for i in range(probes):
        point = rng.integers(1, cells - 1, 3) * spacing
        direction = Direction(i % 3)
        grid.add(
            ImpressedElectricCurrentSource(
                *point, *point, waveform=waveform, direction=direction
            )
        )
        point = rng.integers(1, cells - 1, 3) * spacing
        grid.add(EFieldDetector(*point, *point, direction=direction))
    grid.run(steps, progress=False)
    spectra = [detector.post_process()[1] for detector in grid.detectors]
    power = np.sum(np.abs(spectra) ** 2, axis=0)
    peaks, _ = find_peaks(power)
    return grid.detectors[0].freq[peaks]


def main():
    args = arguments(3)
    bz_path = path(3, args.points, 1e-6)
    pwe = plane_wave_bands(
        bz_path, 1e-6, 0.1e-6, 2.43, args.morpho_grid, args.morpho_order
    )
    betas = bz_path.betas
    jobs = [
        (beta, args.cells, args.steps, args.probes, args.seed)
        for beta in betas.values.T
    ]
    if args.workers == 1:
        results = list(map(simulation, jobs))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            results = list(executor.map(simulation, jobs))
    bands = [
        (distance, frequencies, "3D FDTD")
        for distance, frequencies in zip(betas.cumsum, results)
    ]
    draw(
        bz_path,
        bands,
        1e-6,
        args.output,
        not args.no_plot,
        y_limits=(0, 2),
        curves=pwe,
    )
    print(f"Completed {len(bands)} simulations with seed {args.seed}")


if __name__ == "__main__":
    main()
