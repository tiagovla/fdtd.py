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
        (distance, frequencies, "3D")
        for distance, frequencies in zip(betas.cumsum, results)
    ]
    draw(bz_path, bands, 1e-6, args.output, not args.no_plot)
    print(f"Completed {len(bands)} simulations with seed {args.seed}")


if __name__ == "__main__":
    main()
