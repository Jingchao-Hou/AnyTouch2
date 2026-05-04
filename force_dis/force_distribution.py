import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TRAIN_OBJ_IDS = {6, 41, 52, 53, 59, 69, 70}
TEST_OBJ_IDS = {18, 22, 61}
JSON_PATH = Path("datasets/ToucHD-Force/all_data_direction.json")
OUTPUT_DIR = Path("force_dis")


def extract_obj_id(obj_speed_name):
    obj_name = obj_speed_name.split("_speed")[0]
    return int(obj_name.replace("obj", ""))


def compute_magnitudes(json_data):
    all_magnitudes = []
    train_magnitudes = []
    test_magnitudes = []
    sensor_counts = {}
    sensor_split_values = {
        "digit": {"train": [], "test": []},
        "gelsight": {"train": [], "test": []},
    }

    for obj_speed_name, sensor_data in json_data.items():
        obj_id = extract_obj_id(obj_speed_name)

        for sensor_name, rows in sensor_data.items():
            sensor_counts.setdefault(sensor_name, 0)

            for row in rows:
                if not isinstance(row, list) or len(row) < 4:
                    continue

                fx, fy, fz = row[1], row[2], row[3]
                magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
                all_magnitudes.append(magnitude)
                sensor_counts[sensor_name] += 1

                if obj_id in TRAIN_OBJ_IDS:
                    train_magnitudes.append(magnitude)
                    if sensor_name in sensor_split_values:
                        sensor_split_values[sensor_name]["train"].append(magnitude)
                elif obj_id in TEST_OBJ_IDS:
                    test_magnitudes.append(magnitude)
                    if sensor_name in sensor_split_values:
                        sensor_split_values[sensor_name]["test"].append(magnitude)

    return (
        np.asarray(all_magnitudes, dtype=np.float64),
        np.asarray(train_magnitudes, dtype=np.float64),
        np.asarray(test_magnitudes, dtype=np.float64),
        sensor_counts,
        {
            sensor_name: {
                split_name: np.asarray(split_values, dtype=np.float64)
                for split_name, split_values in split_dict.items()
            }
            for sensor_name, split_dict in sensor_split_values.items()
        },
    )


def build_bin_edges(values, min_bins=20, max_bins=60):
    fd_edges = np.histogram_bin_edges(values, bins="fd")
    if len(fd_edges) - 1 < min_bins or len(fd_edges) - 1 > max_bins:
        return np.histogram_bin_edges(values, bins=max_bins)
    return fd_edges


def summarize(values):
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def format_force_axis(ax, bin_edges):
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    bin_width = float(np.mean(np.diff(bin_edges)))

    ax.set_xlim(x_min, x_max) 
    ax.set_xlabel("Force Magnitude (N)")
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f"{edge:.2f}" for edge in bin_edges], rotation=90, fontsize=7)
    ax.grid(axis="x", which="major", linestyle=":", alpha=0.25)

    ax.text(
        0.99,
        0.98,
        f"Range: {x_min:.2f}-{x_max:.2f} N | Bin width: {bin_width:.2f} N",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def annotate_bar_counts(ax, bars, counts):
    max_count = max(counts) if len(counts) > 0 else 0
    offset = max(max_count * 0.005, 1.0)
    for bar, count in zip(bars, counts):
        if count <= 0:
            continue
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()
        ax.text(
            x,
            y + offset,
            str(int(count)),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6,
        )


def plot_overall(values, bin_edges, output_path):
    counts, edges = np.histogram(values, bins=bin_edges)
    widths = np.diff(edges)

    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(edges[:-1], counts, width=widths, align="edge", color="#2E86AB", edgecolor="black", linewidth=0.5)
    ax.set_title("ToucHD-Force Magnitude Distribution (All Objects, All Sensors)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    format_force_axis(ax, bin_edges)
    annotate_bar_counts(ax, bars, counts)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_train_test(train_values, test_values, bin_edges, output_path, title_prefix=""):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True, sharey=True)
    train_title = "Train Objects Magnitude Distribution"
    test_title = "Test Objects Magnitude Distribution"
    if title_prefix:
        train_title = f"{title_prefix} {train_title}"
        test_title = f"{title_prefix} {test_title}"
    plot_specs = [
        (axes[0], train_values, train_title, "#3FA34D"),
        (axes[1], test_values, test_title, "#E07A5F"),
    ]

    for ax, values, title, color in plot_specs:
        counts, edges = np.histogram(values, bins=bin_edges)
        widths = np.diff(edges)
        bars = ax.bar(edges[:-1], counts, width=widths, align="edge", color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        format_force_axis(ax, bin_edges)
        ax.tick_params(axis="x", labelbottom=True)
        annotate_bar_counts(ax, bars, counts)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    json_path = JSON_PATH
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r") as f:
        json_data = json.load(f)

    all_values, train_values, test_values, sensor_counts, sensor_split_values = compute_magnitudes(json_data)
    if all_values.size == 0:
        raise ValueError("No valid force rows were found in the JSON file.")

    bin_edges = build_bin_edges(all_values)

    plot_overall(
        all_values,
        bin_edges,
        output_dir / "force_magnitude_distribution_all.png",
    )
    plot_train_test(
        train_values,
        test_values,
        bin_edges,
        output_dir / "force_magnitude_distribution_train_test.png",
    )

    for sensor_name in ("digit", "gelsight"):
        sensor_train_values = sensor_split_values[sensor_name]["train"]
        sensor_test_values = sensor_split_values[sensor_name]["test"]
        if sensor_train_values.size == 0 and sensor_test_values.size == 0:
            continue

        sensor_all_values = np.concatenate(
            [values for values in (sensor_train_values, sensor_test_values) if values.size > 0]
        )
        sensor_bin_edges = build_bin_edges(sensor_all_values)
        plot_train_test(
            sensor_train_values,
            sensor_test_values,
            sensor_bin_edges,
            output_dir / f"force_magnitude_distribution_{sensor_name}_train_test.png",
            title_prefix=sensor_name.capitalize(),
        )

    summary = {
        "dataset": str(json_path),
        "train_obj_ids": sorted(TRAIN_OBJ_IDS),
        "test_obj_ids": sorted(TEST_OBJ_IDS),
        "bin_count": int(len(bin_edges) - 1),
        "bin_min": float(bin_edges[0]),
        "bin_max": float(bin_edges[-1]),
        "bin_widths_preview": [float(x) for x in np.diff(bin_edges[: min(len(bin_edges), 6)])],
        "sensor_sample_counts": sensor_counts,
        # "all": summarize(all_values),
        "train": summarize(train_values),
        "test": summarize(test_values),
        "digit": {
            "train": summarize(sensor_split_values["digit"]["train"]),
            "test": summarize(sensor_split_values["digit"]["test"]),
        },
        "gelsight": {
            "train": summarize(sensor_split_values["gelsight"]["train"]),
            "test": summarize(sensor_split_values["gelsight"]["test"]),
        },
    }

    summary_path = output_dir / "force_magnitude_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved overall plot to: {output_dir / 'force_magnitude_distribution_all.png'}")
    print(f"Saved train/test plot to: {output_dir / 'force_magnitude_distribution_train_test.png'}")
    print(f"Saved digit train/test plot to: {output_dir / 'force_magnitude_distribution_digit_train_test.png'}")
    print(f"Saved gelsight train/test plot to: {output_dir / 'force_magnitude_distribution_gelsight_train_test.png'}")
    print(f"Saved summary to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
