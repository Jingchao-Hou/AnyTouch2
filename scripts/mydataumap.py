import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot  


feature_path = "log/mydataset_start/cls_features.npy"
meta_path = "log/mydataset_start/metadata.csv"
out_path = "log/mydataset_start/umap_cls_sensor.png"
movement_out_path = "log/mydataset_start/umap_cls_sensor_movement.png"

frame14_feature_path = "log/mydataset_start_frame14/cls_features.npy"
frame14_meta_path = "log/mydataset_start_frame14/metadata.csv"
frame14_movement_out_path = "log/mydataset_start_frame14/umap_cls_sensor_movement.png"

ordered_feature_path = "log/mydataset_ordered_15_16/cls_features_right.npy"
ordered_meta_path = "log/mydataset_ordered_15_16/metadata.csv"
ordered_movement_out_path = "log/mydataset_ordered_15_16/umap_cls_sensor_movement_right.png"


colors = {
    "digit": "green",
    "gelsight": "red",
}

movement_colors = {
    "center": "tab:blue",
    "right": "tab:orange",
    "left": "tab:green",
    "down": "tab:red",
    "up": "tab:purple",
}

sensor_markers = {
    "digit": "^",
    "gelsight": "s",
}

step_colors = {
    "15": "tab:blue",
    "16": "tab:orange",
    "17": "tab:green",
    "18": "tab:red",
}

target_movements = ["center", "right", "left", "down", "up"]


def load_metadata(meta_path):
    labels = []
    movements = []
    sequences = []
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["sensor"])
            movements.append(row["movement"])
            sequences.append(row["sequence"])
    return labels, movements, sequences


X = np.load(feature_path)
print(f"shape for the full dataset latent: {X.shape}")
labels, movements, _ = load_metadata(meta_path)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
X_2d = umap.UMAP(
    n_components=2,
    random_state=0,
    n_neighbors=100,
).fit_transform(X)

plt.figure(figsize=(8, 6))
for sensor in sorted(set(labels)):
    idx = [i for i, sensor_label in enumerate(labels) if sensor_label == sensor]
    plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        s=5,
        c=colors.get(sensor, "gray"),
        label=sensor,
        alpha=0.8,
    )
plt.legend()
plt.title("UMAP of full dataset ")
plt.tight_layout()
plt.savefig(out_path)
print(f"Saved to {out_path}")

plt.figure(figsize=(8, 6))
for sensor in sorted(set(labels)):
    for movement in target_movements:
        idx = []
        for i, (sensor_label, movement_label) in enumerate(zip(labels, movements)):
            if sensor_label == sensor and movement_label == movement:
                idx.append(i)
        if not idx:
            continue
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=12,
            c=movement_colors.get(movement, "gray"),
            marker=sensor_markers.get(sensor, "o"),
            label=f"{sensor}-{movement}",
            alpha=0.8,
        )
plt.legend(ncol=2, fontsize=8)
plt.title("UMAP of full dataset by Sensor and Movement")
plt.tight_layout()
plt.savefig(movement_out_path)
print(f"Saved to {movement_out_path}")


#########################################################
X_frame14 = np.load(frame14_feature_path)
print("\n shape of only frame14 {X_frame14.shape} \n")
frame14_labels, frame14_movements, _ = load_metadata(frame14_meta_path)
X_frame14_2d = umap.UMAP(
    n_components=2,
    random_state=0,
    n_neighbors=4
).fit_transform(X_frame14)

plt.figure(figsize=(8, 6))
for sensor in sorted(set(frame14_labels)):
    for movement in target_movements:
        idx = []
        for i, (sensor_label, movement_label) in enumerate(
            zip(frame14_labels, frame14_movements)
        ):
            if sensor_label == sensor and movement_label == movement:
                idx.append(i)
        if not idx:
            continue
        plt.scatter(
            X_frame14_2d[idx, 0],
            X_frame14_2d[idx, 1],
            s=28,
            c=movement_colors.get(movement, "gray"),
            marker=sensor_markers.get(sensor, "o"),
            label=f"{sensor}-{movement}",
            alpha=0.8,
        )
plt.legend(ncol=2, fontsize=8)
plt.title("UMAP of Step-14 CLS Features by Sensor and Movement")
plt.tight_layout()
plt.savefig(frame14_movement_out_path)
print(f"Saved to {frame14_movement_out_path}")


##################################################################
X_ordered = np.load(ordered_feature_path)
print("\n shape of ordered {X_ordered.shape} \n")
ordered_labels, _, ordered_sequences = load_metadata(ordered_meta_path)
ordered_steps = [sequence.split("_")[-1] for sequence in ordered_sequences]
X_ordered_2d = umap.UMAP(
    n_components=2,
    random_state=0,
    n_neighbors=4,
).fit_transform(X_ordered)

plt.figure(figsize=(8, 6))
for sensor in sorted(set(ordered_labels)):
    for step in ["15", "16", "17", "18"]:
        idx = []
        for i, (sensor_label, step_label) in enumerate(
            zip(ordered_labels, ordered_steps)
        ):
            if sensor_label == sensor and step_label == step:
                idx.append(i)
        if not idx:
            continue
        plt.scatter(
            X_ordered_2d[idx, 0],
            X_ordered_2d[idx, 1],
            s=40,
            c=step_colors.get(step, "gray"),
            marker=sensor_markers.get(sensor, "o"),
            label=f"{sensor}-step{step}",
            alpha=0.8,
        )
plt.legend(ncol=2, fontsize=7)
plt.title("UMAP of Ordered  by Sensor and Step")
plt.tight_layout()
plt.savefig(ordered_movement_out_path)
print(f"Saved to {ordered_movement_out_path}")
