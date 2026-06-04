import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


COMMON_CMAP_NAME = "YlGnBu"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_class_rows(metrics_path: Path) -> tuple[list[dict[str, float]], float]:
    metrics_json = load_json(metrics_path)
    metrics = metrics_json["tag_class_metrics"]
    acc1 = float(metrics_json["acc1"])
    class_count = len(metrics)
    rows = []

    for class_id in range(class_count):
        class_metrics = metrics[str(class_id)]
        rows.append(
            {
                "class_id": class_id,
                "recall": float(class_metrics["recall"]),
                "support": int(class_metrics["support"]),
            }
        )

    return rows, acc1


def plot_class_accuracy(rows: list[dict[str, float]], acc1: float, output_path: Path) -> None:
    accuracies = [float(row["recall"]) for row in rows]
    cmap = plt.get_cmap(COMMON_CMAP_NAME)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colors = [cmap(norm(accuracy)) for accuracy in accuracies]

    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    bars = ax.bar(
        range(len(rows)),
        accuracies,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("tag class accuracy--training-last-checkpoint")
    ax.set_xlabel("Tag class")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"class {int(row['class_id'])}" for row in rows], rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.text(
        0.98,
        0.98,
        f"acc1-last checkpoint {acc1:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            (
                f"Accuracy {float(row['recall']):.3f}\n"
                f"Count {int(row['support'])}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, pad=0.01)
    colorbar.set_label("Accuracy")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    metrics_path = (
        project_dir
        / "log"
        / "tag-2026-06-04_17-06-30"
        / "tag_class_metrics49.json"
    )
    output_path = project_dir / "datadis" / "tag" / "tag_class_accuracy.png"

    rows, acc1 = load_class_rows(metrics_path)
    plot_class_accuracy(rows, acc1, output_path)


if __name__ == "__main__":
    main()
