import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


TEXTILE_TYPE_INDEX = 10
TEXTILE_TYPE_NAME = "Taxtile type"
TEXTILE_TYPE_CLASSES = 20


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_distribution(split_data, cloth_metadata):
    distribution = defaultdict(int)

    for sample in split_data.values():
        cloth_index = str(sample["cloth_index"])
        if cloth_index not in cloth_metadata:
            continue

        attributes = cloth_metadata[cloth_index]
        textile_type = attributes[TEXTILE_TYPE_INDEX]
        distribution[textile_type] += 1

    return distribution


def build_plot_data(distribution):
    labels = []
    counts = []
    colors = plt.get_cmap("tab20").colors

    for class_idx in range(TEXTILE_TYPE_CLASSES):
        labels.append(f"{TEXTILE_TYPE_NAME}\n{class_idx}")
        counts.append(distribution.get(class_idx, 0))

    legend_handles = [
        Patch(facecolor=colors[class_idx % len(colors)], label=f"Type {class_idx}")
        for class_idx in range(TEXTILE_TYPE_CLASSES)
    ]
    bar_colors = [colors[class_idx % len(colors)] for class_idx in range(TEXTILE_TYPE_CLASSES)]

    return labels, counts, bar_colors, legend_handles


def plot_distribution(ax, distribution, title, highlight_missing=False):
    labels, counts, colors, legend_handles = build_plot_data(distribution)
    x_positions = range(len(labels))
    total_count = sum(counts)
    max_count = max(counts) if counts else 0

    bars = ax.bar(x_positions, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Sample count")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=legend_handles,
        title="Textile types",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.text(
        0.02,
        0.98,
        "GelSight",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    y_offset = max_count * 0.01 if counts else 0
    for bar, count in zip(bars, counts):
        if count == 0:
            continue

        percentage = (count / total_count) * 100 if total_count else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{count}\n{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if highlight_missing:
        highlight_height = max(max_count * 0.08, 1)
        for bar, count in zip(bars, counts):
            if count != 0:
                continue

            rect = Rectangle(
                (bar.get_x(), 0),
                bar.get_width(),
                highlight_height,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=2,
            )
            ax.add_patch(rect)

        ax.set_ylim(top=max_count * 1.18 if max_count else highlight_height * 2)


def main():
    data_dir = Path(__file__).resolve().parents[1]
    train_path = data_dir / "train_data_new.json"
    test_path = data_dir / "test_data_new.json"
    cloth_metadata_path = data_dir / "cloth_metadata.json"

    train_data = load_json(train_path)
    test_data = load_json(test_path)
    cloth_metadata = load_json(cloth_metadata_path)

    train_distribution = compute_distribution(train_data, cloth_metadata)
    test_distribution = compute_distribution(test_data, cloth_metadata)

    fig, axes = plt.subplots(2, 1, figsize=(24, 14), constrained_layout=True)
    plot_distribution(axes[0], train_distribution, "Train Textile Type Distribution")
    plot_distribution(
        axes[1],
        test_distribution,
        "Test Textile Type Distribution",
        highlight_missing=True,
    )
    axes[1].set_xlabel("Encoded textile type")

    output_path = Path(__file__).resolve().parent / "cloth_data_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
