import json
from pathlib import Path

import numpy as np
from PIL import Image


data_root = Path("mydataset")

sum_rgb = np.zeros(3, dtype=np.float64)
sum_sq_rgb = np.zeros(3, dtype=np.float64)
pixel_count = 0

sequence_count = 0
frame_count = 0

for force_json_path in sorted(data_root.glob("*/*/force.json")):
    sequence_count += 1

    with open(force_json_path, "r") as f:
        record = json.load(f)

    sequence_dir = force_json_path.parent
    frames = record.get("frames", [])

    for frame in frames:
        image_path = sequence_dir / frame["frame_file"]
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image, dtype=np.float64) / 255.0

        flat = image.reshape(-1, 3)
        sum_rgb += flat.sum(axis=0)
        sum_sq_rgb += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]
        frame_count += 1

if pixel_count == 0:
    raise RuntimeError("No frames were processed. Check data_root.")

mean = sum_rgb / pixel_count
var = sum_sq_rgb / pixel_count - mean ** 2
std = np.sqrt(np.clip(var, 0.0, None))

print("Sequences scanned:", sequence_count)
print("Frames processed:", frame_count)
print("mean:", mean.tolist())
print("std:", std.tolist())
