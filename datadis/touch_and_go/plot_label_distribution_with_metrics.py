import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


CLASS_COUNT = 20
COMMON_CMAP_NAME = "YlGnBu"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_category_mapping(reference_path: Path) -> dict[int, str]:
    mapping = {}

    with reference_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or ":" not in line:
                continue

            name_part, id_part = line.rsplit(":", 1)
            match = re.search(r"-?\d+", id_part)
            if match is None:
                continue

            mapping[int(match.group())] = name_part.strip().strip("'").strip()

    return mapping


def get_category_mapping(project_dir: Path) -> dict[int, str]:
    candidate_paths = [
        project_dir / "datasets" / "touch_and_go" / "category_reference.txt",
        project_dir.parent / "thesis" / "AnyTouch2" / "datasets" / "touch_and_go" / "category_reference.txt",
        Path("/fastwork/jhou/AnyTouch2/datasets/touch_and_go/category_reference.txt"),
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return load_category_mapping(candidate_path)

    return {}


def load_metric_rows(metrics_path: Path, category_mapping: dict[int, str]) -> list[dict[str, float]]:
    metrics = load_json(metrics_path)
    rows = []

    for class_id in range(CLASS_COUNT):
        class_metrics = metrics[str(class_id)]
        rows.append(
            {
                "class_id": class_id,
                "label": category_mapping.get(class_id, f"Label {class_id}"),
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
    y_offset = max_support * 0.012 if max_support else 1

    cmap = plt.get_cmap(COMMON_CMAP_NAME)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colors = [cmap(norm(f1_score)) for f1_score in f1_scores]

    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=True)
    bars = ax.bar(
        range(len(rows)),
        supports,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title("Touch-and-Go Test Label Distribution With Precision / Recall / F1")
    ax.set_xlabel("Label")
    ax.set_ylabel("Support")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"] for row in rows], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max_support * 1.18 if max_support else 1)

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
    metrics_path = project_dir / "log" / "tag_class_metrics49.json"
    output_path = project_dir / "datadis" / "touch_and_go" / "label_distribution_with_metrics.png"

    category_mapping = get_category_mapping(project_dir)
    rows = load_metric_rows(metrics_path, category_mapping)
    plot_distribution_with_metrics(rows, output_path)


if __name__ == "__main__":
    main()
