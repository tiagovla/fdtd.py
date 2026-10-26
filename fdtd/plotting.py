"""Optional plotting support."""

from itertools import product

import numpy as np


def pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires the 'plot' extra: pip install 'fdtd.py[plot]'."
        ) from error
    return plt


def _plot_box(ax, bounds, color, *, alpha=0.3, outline=False):
    """Draw volumes, sheets, lines, and points without duplicate flat faces."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lower = np.array((bounds.x_min, bounds.y_min, bounds.z_min))
    upper = np.array((bounds.x_max, bounds.y_max, bounds.z_max))
    active = np.flatnonzero(upper > lower)
    if len(active) < 2:
        if len(active):
            ax.plot(*np.stack((lower, upper)).T, color=color, linewidth=2)
        else:
            ax.scatter(*lower, color=color, s=25, depthshade=False)
        return
    corners = np.array(list(product(*zip(lower, upper))))
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    if len(active) == 2:
        normal = next(axis for axis in range(3) if axis not in active)
        faces = [faces[2 * normal]]
    ax.add_collection3d(
        Poly3DCollection(
            [corners[face] for face in faces],
            facecolors="none" if outline else color,
            edgecolors=color,
            alpha=0.8 if outline else alpha,
            linewidths=1,
            linestyles="dashed" if outline else "solid",
        )
    )


def plot_geometry(grid, z_view=False, *, ax=None, show=True):
    """Draw registered objects and connections without preparing the solver.

    Object material colors are preserved. Sources are green, lumped elements
    orange, and detectors purple outlines. Connections follow each element's
    declared direction; sheet and volume extents remain visible.

    Coordinates and returned artists stay in meters. Tick labels use a shared
    display unit (m, mm, µm, or nm) chosen from the domain size. Material-grid
    samples are omitted.
    """
    plt = pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    from .objects import Brick
    from .plane_wave import ScatteredFieldSource, TFSFSource

    if ax is None:
        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    handles = []
    for obj in grid.objects:
        color = obj.material.color
        if isinstance(obj, Brick):
            _plot_box(ax, obj.bounding_box, color)
        else:
            obj.plot_3d(ax, alpha=0.3)
        handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.4, label=obj.name)
        )

    for elements, color, marker, detector in (
        (grid.sources, "#15803d", "o", False),
        (grid.lumped_elements, "#c2670a", "s", False),
        (grid.detectors, "#7e22ce", "D", True),
    ):
        for element in elements:
            bounds = element.bounding_box
            if isinstance(element, ScatteredFieldSource):
                handles.append(
                    Line2D([], [], color=color, label=f"{element.name} (global)")
                )
                continue
            if isinstance(element, TFSFSource):
                _plot_box(ax, bounds, color, outline=True)
                handles.append(
                    Patch(facecolor="none", edgecolor=color, label=element.name)
                )
                continue
            _plot_box(ax, bounds, color, alpha=0.12, outline=detector)
            center = np.array(
                [
                    (bounds.x_min + bounds.x_max) / 2,
                    (bounds.y_min + bounds.y_max) / 2,
                    (bounds.z_min + bounds.z_max) / 2,
                ]
            )
            endpoints = np.tile(center, (2, 1))
            axis = element.direction.value
            endpoints[:, axis] = (
                (bounds.x_min, bounds.y_min, bounds.z_min)[axis],
                (bounds.x_max, bounds.y_max, bounds.z_max)[axis],
            )
            style = {
                "color": color,
                "linestyle": "--" if detector else "-",
                "linewidth": 2,
            }
            ax.plot(*endpoints.T, **style)
            ax.plot(
                *center,
                marker=marker,
                markersize=7,
                markerfacecolor="white" if detector else color,
                markeredgecolor=color,
                linestyle="none",
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    **style,
                    marker=marker,
                    markerfacecolor="white" if detector else color,
                    label=element.name,
                )
            )

    extent = np.array(grid.shape) * grid.spacing
    for unit, scale in (("m", 1), ("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9)):
        if extent.max() >= scale or unit == "nm":
            break
    for name, axis, length in zip("xyz", (ax.xaxis, ax.yaxis, ax.zaxis), extent):
        getattr(ax, f"set_{name}lim")(0, length)
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.set_major_formatter(
            FuncFormatter(lambda value, position: f"{value / scale:g}")
        )
        getattr(ax, f"set_{name}label")(f"{name} [{unit}]", fontsize=11, labelpad=10)
        axis.pane.set_facecolor("#f1f5f9")
        axis.pane.set_edgecolor("#d9e2ec")
        axis.line.set_color("#94a3b8")
    ax.tick_params(labelsize=9, pad=2)
    ax.set_box_aspect(extent)
    ax.set_proj_type("ortho")
    ax.grid(False)
    ax.view_init(elev=90 if z_view else 23, azim=-90 if z_view else -55)
    ax.set_title("Simulation geometry", fontsize=14, pad=18)
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
            fontsize=9,
        )
    if show:
        plt.show()
    return ax


def visualize_grid(grid, **kwargs):
    """Compatibility wrapper for Grid.plot_geometry()."""
    return grid.plot_geometry(**kwargs)
