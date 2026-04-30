from pathlib import Path
import json

import matplotlib.pyplot as plt


SENSORS = ["digit", "biotip", "gelsight", "duragel", "dm"]


def load_used_counts(json_path: Path) -> tuple[dict[str, int], int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    counts = {sensor: 0 for sensor in SENSORS}

    for session_name, session_data in data.items():
        if not isinstance(session_data, dict):
            raise ValueError(f"Unexpected format for session {session_name}")

        for sensor in SENSORS:
            sensor_items = session_data.get(sensor, [])
            if not isinstance(sensor_items, list):
                raise ValueError(
                    f"Unexpected data format for sensor {sensor} in session {session_name}"
                )
            counts[sensor] += len(sensor_items)

    return counts, len(data)


def count_raw_files(dataset_dir: Path) -> dict[str, int]:
    raw_counts = {sensor: 0 for sensor in SENSORS}

    for session_dir in dataset_dir.iterdir():
        if not session_dir.is_dir() or not session_dir.name.startswith("obj"):
            continue

        for sensor in SENSORS:
            sensor_dir = session_dir / sensor
            if sensor_dir.exists():
                raw_counts[sensor] += sum(1 for path in sensor_dir.iterdir() if path.is_file())

    return raw_counts


def plot_used_distribution(
    used_counts: dict[str, int],
    raw_counts: dict[str, int],
    session_count: int,
    output_path: Path,
) -> None:
    values = [used_counts[sensor] for sensor in SENSORS]
    total_used = sum(values)
    percentages = [
        (value / total_used * 100) if total_used else 0.0
        for value in values
    ]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(SENSORS, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"])
    plt.ylabel("Number of used samples")
    plt.xlabel("Sensor")
    plt.title(f"ToucHD-Force Sensor Distribution from all_data_direction.json\n{session_count} sessions")

    for bar, sensor, value, percentage in zip(bars, SENSORS, values, percentages):
        extra_note = ""
        if sensor == "dm" and value == 0 and raw_counts.get("dm", 0) > 0:
            extra_note = f"\nraw: {raw_counts['dm']}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n{percentage:.1f}%{extra_note}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.figtext(
        0.5,
        0.01,
        "Note: 'dm' exists in the dataset folders but is not present in all_data_direction.json, so its used count is 0.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "all_data_direction.json"
    output_path = base_dir / "sensor_distribution.png"

    used_counts, session_count = load_used_counts(json_path)
    raw_counts = count_raw_files(base_dir)
    plot_used_distribution(used_counts, raw_counts, session_count, output_path)

    total_used = sum(used_counts.values())
    print(f"Saved plot to: {output_path}")
    print(f"Sessions in JSON: {session_count}")
    print(f"Total used samples in JSON: {total_used}")
    for sensor in SENSORS:
        percentage = (used_counts[sensor] / total_used * 100) if total_used else 0.0
        print(
            f"{sensor}: used={used_counts[sensor]}, percentage={percentage:.2f}%, raw_files={raw_counts[sensor]}"
        )


if __name__ == "__main__":
    main()
