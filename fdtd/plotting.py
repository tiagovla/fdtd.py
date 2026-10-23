"""Optional plotting support."""


def pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting requires the 'plot' extra: pip install 'fdtd.py[plot]'."
        ) from error
    return plt


def visualize_grid(grid):
    return grid.plot_3d()
