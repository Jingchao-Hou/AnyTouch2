from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt


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
                raise ValueError(f"Could not parse category id from: {line}")

            category_name = name_part.strip().strip("'")
            category_id = int(match.group())
            mapping[category_id] = category_name

    return mapping


def load_label_counts(label_path: Path) -> Counter:
    counts = Counter()

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Invalid label format on line {line_number}: {line}")

            try:
                label_id = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label id on line {line_number}: {line}"
                ) from exc

            counts[label_id] += 1

    return counts


def plot_distribution(ax, counts: Counter, category_mapping: dict[int, str], title: str, color: str) -> None:
    label_ids = sorted(set(category_mapping) | set(counts))
    label_names = [category_mapping.get(label_id, f"Unknown ({label_id})") for label_id in label_ids]
    frequencies = [counts.get(label_id, 0) for label_id in label_ids]
    total_count = sum(frequencies)
    max_count = max(frequencies) if frequencies else 0

    bars = ax.bar(range(len(label_ids)), frequencies, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(label_ids)))
    ax.set_xticklabels(
        [f"{name}\n({label_id})" for name, label_id in zip(label_names, label_ids)],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    y_offset = max_count * 0.01 if max_count else 0
    for bar, frequency in zip(bars, frequencies):
        if frequency == 0:
            continue

        percentage = (frequency / total_count) * 100 if total_count else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{frequency}\n{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parents[1]

    train_path = project_dir / "data" / "train_bg.txt"
    test_path = project_dir / "data" / "test_bg.txt"
    reference_path = base_dir / "category_reference.txt"
    output_path = base_dir / "train_test_bg_distribution.png"

    category_mapping = load_category_mapping(reference_path)
    train_counts = load_label_counts(train_path)
    test_counts = load_label_counts(test_path)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), constrained_layout=True)
    plot_distribution(axes[0], train_counts, category_mapping, "Train BG Label Distribution", "#4C78A8")
    plot_distribution(axes[1], test_counts, category_mapping, "Test BG Label Distribution", "#F58518")
    axes[1].set_xlabel("Label")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
