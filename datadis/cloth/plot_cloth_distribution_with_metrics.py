import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


TEXTILE_TYPE_COUNT = 20
COMMON_CMAP_NAME = "YlGnBu"
TEXTILE_TYPE_NAMES = [
    "broadcloth",
    "cotton",
    "denim",
    "fleece",
    "hairy",
    "leather",
    "other",
    "polyester",
    "suit",
    "wool",
    "corduroy",
    "crepe",
    "flannel",
    "garbardine",
    "knit",
    "net",
    "parka",
    "satin",
    "velvet",
    "woven",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_metric_rows(metrics_path: Path) -> list[dict[str, float]]:
    metrics = load_json(metrics_path)
    rows = []

    for class_id in range(TEXTILE_TYPE_COUNT):
        class_metrics = metrics[str(class_id)]
        rows.append(
            {
                "class_id": class_id,
                "label": TEXTILE_TYPE_NAMES[class_id],
                "precision": float(class_metrics["precision"]),
                "recall": float(class_metrics["recall"]),
                "f1": float(class_metrics["f1"]),
                "support": int(class_metrics["support"]),
            }
        )

    return rows


def plot_distribution_with_metrics(rows: list[dict[str, float]], output_path: Path) -> None:
    supports = [row["support"] for row in rows]
    f1_scores = [row["f1"] for row in rows]
    max_support = max(supports) if supports else 0
    y_offset = max_support * 0.015 if max_support else 1

    cmap = plt.get_cmap(COMMON_CMAP_NAME)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colors = [cmap(norm(f1_score)) for f1_score in f1_scores]

    fig, ax = plt.subplots(figsize=(22, 9), constrained_layout=True)
    bars = ax.bar(
        range(len(rows)),
        supports,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("Cloth Textile Type Test Distribution With Precision / Recall / F1")
    ax.set_xlabel("Textile type")
    ax.set_ylabel("Support")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max_support * 1.28 if max_support else 1)

    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            (
                f"S {row['support']}\n"
                f"P {row['precision']:.3f}\n"
                f"R {row['recall']:.3f}\n"
                f"F1 {row['f1']:.3f}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, pad=0.01)
    colorbar.set_label("F1 score")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    metrics_path = project_dir / "log" / "cloth_class_metrics_epoch_49.json"
    output_path = project_dir / "datadis" / "cloth" / "cloth_data_distribution_with_metrics.png"

    rows = load_metric_rows(metrics_path)
    plot_distribution_with_metrics(rows, output_path)


if __name__ == "__main__":
    main()
