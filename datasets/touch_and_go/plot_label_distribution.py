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

            try:
                _, label_text = line.rsplit(",", 1)
                label_id = int(label_text.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label format on line {line_number}: {line}"
                ) from exc

            counts[label_id] += 1

    return counts


def plot_distribution(counts: Counter, category_mapping: dict[int, str], output_path: Path) -> None:
    label_ids = sorted(set(category_mapping) | set(counts))
    label_names = [category_mapping.get(label_id, f"Unknown ({label_id})") for label_id in label_ids]
    frequencies = [counts.get(label_id, 0) for label_id in label_ids]

    plt.figure(figsize=(16, 8))
    bars = plt.bar(range(len(label_ids)), frequencies, color="#4C78A8")
    plt.xticks(range(len(label_ids)), [f"{name}\n({label_id})" for name, label_id in zip(label_names, label_ids)], rotation=45, ha="right")
    plt.ylabel("Number of samples")
    plt.xlabel("Label")
    plt.title("Touch and Go Label Distribution")

    for bar, frequency in zip(bars, frequencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(frequency),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    label_path = base_dir / "label.txt"
    reference_path = base_dir / "category_reference.txt"
    output_path = base_dir / "label_distribution.png"

    category_mapping = load_category_mapping(reference_path)
    counts = load_label_counts(label_path)
    plot_distribution(counts, category_mapping, output_path)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
