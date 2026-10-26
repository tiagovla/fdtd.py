"""Compare scattered-field and TF/SF illumination of a dielectric block.

 y
 ^
79 +##################################################+
   #                 CPML absorber                    #
   #                                                  #
69 #     + - - - - - - - - - - - - - - - - - - +      #
   #    |          TOTAL-FIELD REGION           |     #
   #                                                  #
   #    |   Incident Gaussian pulse             |     #
   #          --->                                    #
44 #    |     --->      +----------+            |     #
   #          --->      |          |                  #
   #    |     --->      | eps_r=4  |            |     #
   #          --->      | mu_r=1   |                  #
36 #    |               +----------+            |     #
   #                                                  #
   #    |                                       |     #
10 #     + - - - - - - - - - - - - - - - - - - +      #
   #          SCATTERED-FIELD REGION                  #
   #                 CPML absorber                    #
 0 +##################################################+--> x
   0   10                36         44          69    79
"""

import argparse
from pathlib import Path

import numpy as np

from fdtd import (
    Brick,
    CPMLBoundary,
    GaussianWaveform,
    Grid,
    Material,
    PlaneWave,
    ScatteredFieldSource,
    TFSFSource,
)
from fdtd.constants import SPEED_LIGHT
from fdtd.plotting import pyplot

BACKGROUND = Material("plane-wave-background")
DIELECTRIC = Material("plane-wave-dielectric", eps_r=4)


def simulate(cells, steps, formulation, *, progress=True):
    grid = Grid((cells, cells, 1), spacing=1e-3)
    pml = 8
    grid.add(CPMLBoundary(pml))
    center = np.array((cells // 2, cells // 2, 0.5)) * grid.spacing
    half_size = np.array((4, 4, 0.5)) * grid.spacing
    grid.add(Brick(*(center - half_size), *(center + half_size), material=DIELECTRIC))
    tau = 10 * grid.dx / SPEED_LIGHT
    wave = PlaneWave(GaussianWaveform(5 * tau, tau), material=BACKGROUND)
    lower, upper = (pml + 2) * grid.dx, (cells - pml - 3) * grid.dx
    source = (
        ScatteredFieldSource(wave)
        if formulation == "scattered"
        else TFSFSource(lower, lower, 0, upper, upper, 0, wave=wave)
    )
    grid.add(source)
    grid.run(steps, progress=progress)
    grid.plot_disc3()
    x, y = np.meshgrid(grid.x, grid.y, indexing="ij")
    positions = np.stack((x, y, np.zeros_like(x)), axis=-1)
    incident = wave.electric_field(positions, grid.time_passed)[..., 2]
    stored = grid.E[:, :, 0, 2]
    total_region = (
        (x >= lower) & (x < upper) & (y >= lower) & (y < upper)
        if formulation == "tfsf"
        else np.zeros(x.shape, dtype=bool)
    )
    scattered = stored - np.where(total_region, incident, 0)
    total = scattered + incident
    return grid, total, scattered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", choices=("scattered", "tfsf"), default="tfsf")
    parser.add_argument("--cells", type=int, default=80)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells < 40:
        parser.error("--cells must be at least 40")
    if args.steps < 1:
        parser.error("--steps must be positive")
    grid, total, scattered = simulate(
        args.cells, args.steps, args.formulation, progress=not args.no_plot
    )
    if not args.no_plot or args.output:
        plt = pyplot()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, field, title in zip(
            axes, (total, scattered), ("Total Ez", "Scattered Ez")
        ):
            limit = np.max(abs(field)) or 1
            image = ax.imshow(
                field.real.T,
                origin="lower",
                extent=(0, grid.x[-1] * 1e3, 0, grid.y[-1] * 1e3),
                vmin=-limit,
                vmax=limit,
                cmap="bwr",
            )
            ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]")
            fig.colorbar(image, ax=ax, label="V/m")
        if args.output:
            fig.savefig(args.output, dpi=160)
        if not args.no_plot:
            plt.show()
        plt.close(fig)
    print(
        f"Completed {grid.current_time_step} steps ({args.formulation}); "
        f"peak scattered Ez = {np.max(abs(scattered)):.6g} V/m"
    )


if __name__ == "__main__":
    main()
