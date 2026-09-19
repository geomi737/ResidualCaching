"""Plot this workspace's local metrics into its figures directory."""

from pathlib import Path


WORKSPACE = Path(__file__).parent
RESULTS = WORKSPACE / "results"
FIGURES = WORKSPACE / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    raise SystemExit("Implement a plotter that reads only this workspace's results directory.")


if __name__ == "__main__":
    main()
