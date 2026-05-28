import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

save_dir = Path("depth_visual")
save_dir.mkdir(exist_ok=True)

# find depth files
files = sorted(
    Path(".").glob("depth_*.npy")
)

for file in files:

    img = np.load(file)

    print(file.name)
    # print("Shape:", img.shape)
    # print("Data type:", img.dtype)
    # print("Min:", img.min())
    # print("Max:", img.max())
    # print("Dimensions:", img.ndim)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.colorbar()
    plt.title("Depth")
    out = save_dir / f"{file.stem}.png"
    plt.savefig(out)
    plt.close()