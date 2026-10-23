import subprocess
import sys


def test_core_import_and_simulation_do_not_load_matplotlib():
    result = subprocess.run([sys.executable, "-c", """
import sys
import fdtd
grid = fdtd.Grid((4, 1, 1), 1e-3)
grid.run(2, progress=False)
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
