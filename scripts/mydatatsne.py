import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

feature_path = "log/mydataset_start/cls_features.npy"
meta_path = "log/mydataset_start/metadata.csv"
out_path = "log/mydataset_start/tsne_cls_sensor.png"

X = np.load(feature_path)
print(f"The shape of the representation  {X.shape}")

labels = []
with open(meta_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(row["sensor"])

tsne = TSNE(n_components=2,random_state=0)
X_2d = tsne.fit_transform(X)
print(f"The shape after the fit transform {X_2d.shape}")

colors = {
    "digit": "green",
    "gelsight": "red",
}

plt.figure(figsize=(8, 6))
for sensor in ["digit", "gelsight"]:
    idx = [i for i, x in enumerate(labels) if x == sensor]
    plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        s=5,
        c=colors[sensor],
        label=sensor,
        alpha=0.8,
    )

plt.legend()
plt.title("t-SNE of mydataset CLS Features using Anytouch2")
plt.tight_layout()
plt.savefig(out_path)
print(f"Saved to {out_path}")
