import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

feature_path = "log/mydataset_start/cls_features.npy"
meta_path = "log/mydataset_start/metadata.csv"
out_path = "log/mydataset_start/tsne_cls_sensor.png"
movement_out_path = "log/mydataset_start/tsne_cls_sensor_movement.png"
frame14_feature_path = "log/mydataset_start_frame14/cls_features.npy"
frame14_meta_path = "log/mydataset_start_frame14/metadata.csv"
frame14_movement_out_path = "log/mydataset_start_frame14/tsne_cls_sensor_movement.png"
ordered_feature_path = "log/mydataset_ordered_15_16/cls_features_right.npy"
ordered_meta_path = "log/mydataset_ordered_15_16/metadata.csv"
ordered_movement_out_path = "log/mydataset_ordered_15_16/tsne_cls_sensor_movement_right.png"

X = np.load(feature_path)
print(f"The shape of the representation  {X.shape}")

labels = []
movements = []
with open(meta_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row["sensor"])
        movements.append(row["movement"])

Path(out_path).parent.mkdir(parents=True, exist_ok=True)

tsne = TSNE(n_components=2, random_state=0,perplexity=20)
X_2d = tsne.fit_transform(X)
print(f"The shape after the fit transform {X_2d.shape}")

colors = {
    "digit": "green",
    "gelsight": "red",
}

plt.figure(figsize=(8, 6))
for sensor in sorted(set(labels)):
    idx = [i for i, x in enumerate(labels) if x == sensor]
    plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        s=5,
        c=colors.get(sensor, "gray"),
        label=sensor,
        alpha=0.8,
    )

plt.legend()
plt.title("t-SNE of mydataset CLS Features using Anytouch2")
plt.tight_layout()
plt.savefig(out_path)
print(f"Saved to {out_path}")

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

target_movements = ["center", "right", "left", "down", "up"]
###########################
plt.figure(figsize=(8, 6))
for sensor in sorted(set(labels)):
    for movement in target_movements:
        idx = [
            i
            for i, (sensor_label, movement_label) in enumerate(zip(labels, movements))
            if sensor_label == sensor and movement_label == movement
        ]
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
plt.title("t-SNE of mydataset CLS Features by Sensor and Movement")
plt.tight_layout()
plt.savefig(movement_out_path)
print(f"Saved to {movement_out_path}")

X_frame14 = np.load(frame14_feature_path)
print(f"The shape of the step14 representation {X_frame14.shape}")

frame14_labels = []
frame14_movements = []
with open(frame14_meta_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame14_labels.append(row["sensor"])
        frame14_movements.append(row["movement"])

X_frame14_2d = TSNE(n_components=2, random_state=0, perplexity=5).fit_transform(X_frame14)
print(f"The shape after the step14 fit transform {X_frame14_2d.shape}")

plt.figure(figsize=(8, 6))
for sensor in sorted(set(frame14_labels)):
    for movement in target_movements:
        idx = [
            i
            for i, (sensor_label, movement_label) in enumerate(
                zip(frame14_labels, frame14_movements)
            )
            if sensor_label == sensor and movement_label == movement
        ]
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
plt.title("t-SNE of Step-14 CLS Features by Sensor and Movement")
plt.tight_layout()
plt.savefig(frame14_movement_out_path)
print(f"Saved to {frame14_movement_out_path}")

X_ordered = np.load(ordered_feature_path)
print(f"The shape of the ordered-step representation {X_ordered.shape}")

ordered_labels = []
ordered_steps = []
with open(ordered_meta_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ordered_labels.append(row["sensor"])
        ordered_steps.append(row["sequence"].split("_")[-1])

X_ordered_2d = TSNE(n_components=2, random_state=0, perplexity=3).fit_transform(X_ordered)
print(f"The shape after the ordered-step fit transform {X_ordered_2d.shape}")

step_colors = {
    "15": "tab:blue",
    "16": "tab:orange",
}

plt.figure(figsize=(8, 6))
for sensor in sorted(set(ordered_labels)):
    for step in ["15", "16","17","18"]:
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
plt.title("t-SNE of up center down left")
plt.tight_layout()
plt.savefig(ordered_movement_out_path)
print(f"Saved to {ordered_movement_out_path}")
