import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "script, arguments",
    [
        ("voltage_resistor.py", ["--steps", "32"]),
        ("periodic_boundary.py", ["--steps", "32", "--cells", "8"]),
        (
            "photonic_crystal_bandgap_2d.py",
            ["--steps", "32", "--cells", "8", "--points", "1", "--probes", "1"],
        ),
        (
            "photonic_crystal_bandgap_3d.py",
            [
                "--steps",
                "32",
                "--cells",
                "8",
                "--points",
                "1",
                "--probes",
                "1",
                "--workers",
                "2",
            ],
        ),
    ],
)
def test_example_smoke_from_another_directory(script, arguments, tmp_path):
    if "bandgap" in script and (
        importlib.util.find_spec("morpho") is None
        or importlib.util.find_spec("scipy") is None
    ):
        pytest.skip("Bandgap example dependencies are optional")
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *arguments, "--no-plot"],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Completed" in result.stdout
