import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

save_dir = Path("original_visual")
save_dir.mkdir(exist_ok=True)

files = sorted(Path(".").glob("deformation_*.npy"))

for file in files:
    img = np.load(file)
    print(file.name, img.shape)
    dx = img[:, :, 0]
    dy = img[:, :, 1]

    Y, X = np.mgrid[
        0:dx.shape[0],
        0:dx.shape[1]
    ]
    plt.figure(figsize=(8, 8))
    plt.quiver(
        X,
        Y,
        dx,
        dy
    )

    plt.title("Deformation Field")
    out = save_dir / f"{file.stem}_quiver.png"
    plt.savefig(out)

    plt.close()

print("Done.")
# img = np.load("deformation_0.npy")

# dx = img[:,:,0]
# dy = img[:,:,1]

# magnitude = np.sqrt(dx**2 + dy**2)
# angle = np.arctan2(dy, dx)

# plt.figure(figsize=(6,6))
# plt.imshow(angle, cmap="twilight")
# plt.colorbar()
# plt.title("Direction")
# out = save_dir / "deformation_angle.png"
# plt.savefig(out)