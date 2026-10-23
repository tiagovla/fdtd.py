import argparse
from pathlib import Path

from fdtd import Brick, Grid, Material, Resistor, SineWaveform, VoltageDetector, VoltageSource


def main():
    parser = argparse.ArgumentParser(description="Voltage source and resistor simulation")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    Material.load(Path(__file__).with_name("materials.json"))
    grid = Grid((10, 10, 10), 1e-3)
    for element in (
        Brick(4e-3, 2e-3, 8e-3, 6e-3, 8e-3, 8e-3, "PEC"),
        Brick(4e-3, 2e-3, 4e-3, 6e-3, 8e-3, 4e-3, "PEC"),
        VoltageSource(4e-3, 2e-3, 4e-3, 6e-3, 2e-3, 8e-3,
                      waveform=SineWaveform(frequency=1e9), resistance=50),
        Resistor(4e-3, 8e-3, 4e-3, 6e-3, 8e-3, 8e-3, 50),
        VoltageDetector(4e-3, 5e-3, 4e-3, 6e-3, 5e-3, 8e-3, plot=not args.no_plot),
    ):
        grid.add(element)
    grid.run(args.steps, progress=not args.no_plot)
    if not args.no_plot:
        grid.plot_3d()
    print(f"Completed {grid.current_time_step} steps")


if __name__ == "__main__":
    main()
