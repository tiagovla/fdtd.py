import subprocess
import sys

import pytest


def test_core_import_and_simulation_do_not_load_matplotlib():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import fdtd
grid = fdtd.Grid((4, 1, 1), 1e-3)
grid.run(2, progress=False)
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "method", ["plot_planes", "plot_disc", "plot_disc2", "plot_disc3", "plot_3d"]
)
def test_material_plotting_supports_rectangular_grids(method, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fdtd import Grid

    monkeypatch.setattr(plt, "show", lambda: None)
    grid = Grid((3, 4, 2), 1e-3)
    grid.cell_material[:] = (2, 1, 0, 0)
    grid._prepare()
    getattr(grid, method)()
    plt.close("all")
