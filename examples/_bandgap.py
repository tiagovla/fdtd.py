import argparse

import numpy as np

from fdtd import Material
from fdtd.exceptions import MaterialNotFound


def arguments(dimension):
    parser = argparse.ArgumentParser(
        description=f"{dimension}D photonic crystal bandgap"
    )
    parser.add_argument("--cells", type=int, default=24 if dimension == 2 else 16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        args.cells < 8
        or args.steps < 2
        or min(args.points, args.probes, args.workers) < 1
    ):
        parser.error("cells must be at least 8, steps at least 2, and counts positive")
    return args


def dielectric(name, eps_r):
    try:
        return Material[name]
    except MaterialNotFound:
        return Material(name, eps_r=eps_r)


def path(dimension, intervals, period):
    from morpho import BrillouinZonePath, SymmetryPoint

    if dimension == 2:
        points = [((0, 0), "Γ"), ((0.5, 0), "X"), ((0.5, 0.5), "M"), ((0, 0), "Γ")]
    else:
        points = [
            ((0.5, 0.5, 0.5), "D"),
            ((0, 0, 0.5), "Z"),
            ((0, 0, 0), "Γ"),
            ((0, 0, 0.5), "Z"),
            ((0.5, 0, 0.5), "T"),
            ((0.5, 0, 0), "X"),
        ]
    vectors = np.eye(dimension) * period
    return BrillouinZonePath(
        *vectors,
        [SymmetryPoint(p, name) for p, name in points],
        points_per_segment=intervals,
        strategy="linear" if dimension == 2 else "interpolate",
    )


def draw(bz_path, bands, period, output, show, y_limits=None):
    if not output and not show:
        return
    import matplotlib.pyplot as plt

    from fdtd.constants import SPEED_LIGHT

    _, ax = plt.subplots(figsize=(6, 4))
    colors = {"TM": "blue", "TE": "red", "3D": "blue"}
    for distance, frequencies, polarization in bands:
        ax.scatter(
            np.full(len(frequencies), distance),
            frequencies * period / SPEED_LIGHT,
            color=colors.get(polarization, "blue"),
            s=8,
            label=polarization,
        )
    ax.set_xticks(bz_path.point_locations, bz_path.point_names)
    ax.set_xlim(0, bz_path.point_locations[-1])
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xlabel("Bloch wave vector")
    ax.set_ylabel(r"Frequency $\omega a/(2\pi c)$")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys())
    ax.figure.tight_layout()
    if output:
        ax.figure.savefig(output)
    if show:
        plt.show()
    plt.close(ax.figure)
