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
    betas = bz_path.betas
    bands = []
    for magnetic, name in ((False, "TM"), (True, "TE")):
        for beta, distance in zip(betas.values.T, betas.cumsum):
            frequencies = simulation(
                beta, args.cells, args.steps, args.probes, args.seed, magnetic
            )
            bands.append((distance, frequencies, name))
    draw(bz_path, bands, 1e-6, args.output, not args.no_plot, y_limits=(0, 1))
    print(f"Completed {len(bands)} simulations with seed {args.seed}")


if __name__ == "__main__":
    main()
