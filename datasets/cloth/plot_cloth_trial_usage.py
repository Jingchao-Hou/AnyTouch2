import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_trial_usage(split_data):
    trial_usage = defaultdict(set)

    for sample in split_data.values():
        cloth_index = sample["cloth_index"]
        trial_index = sample["trial_index"]
        trial_usage[cloth_index].add(trial_index)

    return {cloth_index: len(trials) for cloth_index, trials in sorted(trial_usage.items())}


def plot_usage(ax, usage_by_cloth, title, color):
    cloth_indices = list(usage_by_cloth.keys())
    counts = list(usage_by_cloth.values())
    total_trials = sum(counts)

    bars = ax.bar(
        range(len(cloth_indices)),
        counts,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        label=title.split()[0],
    )
    ax.set_title(title)
    ax.set_ylabel("Unique trial count")
    ax.set_xticks(range(len(cloth_indices)))
    ax.set_xticklabels(cloth_indices, rotation=90, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
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
    ax.text(
        0.02,
        0.90,
        f"Unique cloth types: {len(cloth_indices)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    y_offset = max(counts) * 0.01 if counts else 0
    for bar, count in zip(bars, counts):
        percentage = (count / total_trials) * 100 if total_trials else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_offset,
            f"{count}\n{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def main():
    data_dir = Path(__file__).resolve().parents[1]
    train_data = load_json(data_dir / "train_data_new.json")
    test_data = load_json(data_dir / "test_data_new.json")

    train_usage = compute_trial_usage(train_data)
    test_usage = compute_trial_usage(test_data)

    fig, axes = plt.subplots(2, 1, figsize=(24, 14), constrained_layout=True)
    plot_usage(axes[0], train_usage, "Train Cloth Usage by Cloth Index", "#4C72B0")
    plot_usage(axes[1], test_usage, "Test Cloth Usage by Cloth Index", "#DD8452")
    axes[1].set_xlabel("Cloth index")

    output_path = Path(__file__).resolve().parent / "cloth_trial_usage_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
