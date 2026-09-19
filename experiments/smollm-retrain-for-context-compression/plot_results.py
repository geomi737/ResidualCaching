"""Create figures only from this experiment's local evaluation metrics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


EXPERIMENT = "smollm-retrain-for-context-compression"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    valid = [row for row in rows if "adaptation_tokens" in row and key in row]
    return [row["adaptation_tokens"] for row in valid], [row[key] for row in valid]


def save_loss(control: list[dict], residual: list[dict], figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, rows, colour in (("Compressed residual", residual, "#6a3d9a"),
                                ("Uncompressed control", control, "#1f78b4")):
        x, y = series(rows, "student_loss")
        if x:
            axis.plot(x, y, marker="o", label=label, color=colour)
    axis.set(title="Validation loss during adaptation", xlabel="Adaptation tokens", ylabel="Student loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "validation_loss.png", dpi=160)
    plt.close(fig)


def save_accuracy(control: list[dict], residual: list[dict], figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, rows, colour in (("Compressed residual", residual, "#6a3d9a"),
                                ("Uncompressed control", control, "#1f78b4")):
        x, y = series(rows, "student_acc")
        if x:
            axis.plot(x, y, marker="o", label=label, color=colour)
    axis.set(title="Validation accuracy during adaptation", xlabel="Adaptation tokens", ylabel="Student accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "validation_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-results", type=Path)
    parser.add_argument("--residual-results", type=Path)
    parser.add_argument("--figures", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    workspace_results = Path(__file__).parent / "results"
    control_results = args.control_results or workspace_results
    residual_results = args.residual_results or workspace_results
    control = load_jsonl(control_results / "eval_metrics_control.jsonl")
    residual = load_jsonl(residual_results / "eval_metrics_residual.jsonl")
    if not control and not residual:
        raise SystemExit(
            f"No local evaluation metrics found in {control_results} or {residual_results}"
        )
    save_loss(control, residual, args.figures)
    save_accuracy(control, residual, args.figures)
    print(f"Wrote {EXPERIMENT} figures to {args.figures}")


if __name__ == "__main__":
    main()
