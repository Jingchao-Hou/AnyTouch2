import json
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_rmse_bar_from_json(json_path, save_path, title=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    force_magnitudes = np.array(data["force_magnitudes"], dtype=float)
    rmse = np.array(data["rmse"], dtype=float)

    bin_edges = np.histogram_bin_edges(force_magnitudes, bins="fd")

    if len(bin_edges) < 3:
        bin_edges = np.linspace(force_magnitudes.min(), force_magnitudes.max(), 6)

    bin_ids = np.clip(
        np.digitize(force_magnitudes, bin_edges) - 1,
        0,
        len(bin_edges) - 2
    )

    bar_centers = []
    bar_widths = []
    avg_rmse = []

    shrink = 0.4

    for i in range(len(bin_edges) - 1):
        mask = bin_ids == i
        if np.any(mask):
            left = bin_edges[i]
            right = bin_edges[i + 1]

            width = (right - left) * shrink
            center = left + width / 2
            mean_rmse = rmse[mask].mean()

            bar_centers.append(center)
            bar_widths.append(width)
            avg_rmse.append(mean_rmse)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        bar_centers,
        avg_rmse,
        width=bar_widths,
        align="center",
        edgecolor="black",
        alpha=0.8
    )
   
    plt.xticks(
        bar_centers,
        [f"{x:.1f}" for x in bar_centers],
        rotation=45
    )

    for bar, value in zip(bars, avg_rmse):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5, 
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90
        )

    plt.xlabel("Force Magnitude (N)")
    plt.ylabel("Average RMSE (mN)")  

    if title:
        plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved figure at {save_path}")


if __name__ == "__main__":
    plot_rmse_bar_from_json(
        "/fastwork/jhou/AnyTouch2/log/touchd_eval-gelsight-2026-05-05_12-17-37/eval_force_rmse_epoch0.json",
        "/fastwork/jhou/AnyTouch2/force_dis/gelsight_rmse.png",
        title="Gelsight: Average RMSE by Force Magnitude"
    )

    plot_rmse_bar_from_json(
        "/fastwork/jhou/AnyTouch2/log/touchd_eval-digit-2026-05-05_12-12-57/eval_force_rmse_epoch0.json",
        "/fastwork/jhou/AnyTouch2/force_dis/digit_rmse.png",
        title="Digit: Average RMSE by Force Magnitude"
    )