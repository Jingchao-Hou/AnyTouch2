import csv
import os
import sys
from pathlib import Path
sys.path.append(os.getcwd())
import copy
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig
from config_probe import parse_args
from model.tactile_mae import TactileVideoMAE

# Anytouch2 normalization
offset = 130.0 / 255.0
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

sensor_name_to_id = {
    "gelsight": 0,
    "digit": 1,
    "gelslim": 2,
    "gelsight_mini": 3,
    "duragel": 4,
    "dm": 5,
    "universal": -1,
}


def random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_model_from_multi_clip(ckpt, model):
    new_ckpt = {}
    for key, item in ckpt.items():
        if "touch_mae_model" in key and "decoder" not in key and "mask_token" not in key:
            new_ckpt[key.replace("touch_mae_model.", "")] = copy.deepcopy(item)

    for key, value in model.named_parameters():
        if key not in new_ckpt:
            new_ckpt[key] = value

    model.load_state_dict(new_ckpt, strict=True)
    return model


def build_model(args):
    if args.model_size != "base":
        raise NotImplementedError("Only model_size=base is supported.")
    if args.model != "anytouch":
        raise NotImplementedError(f"Model {args.model} is not implemented.")

    config = AutoConfig.from_pretrained("CLIP-B-16/config.json")
    model = TactileVideoMAE(args, config, args.num_frames, 1)
    ckpt = torch.load(args.load_path, map_location="cpu")
    return load_model_from_multi_clip(ckpt, model)


def data_loader(data_root, sensors):
    dataset = []
    for sensor in sensors:
        sensor_dir = Path(data_root) / sensor
        for clip_dir in sorted(path for path in sensor_dir.iterdir() if path.is_dir()):
            frames_dir = clip_dir / "video_frames"
            frame_paths = sorted(frames_dir.glob("*.png"))
            if frame_paths:
                dataset.append((sensor, clip_dir, frame_paths))

    return dataset


def load_sensor_backgrounds(data_root, sensors):
    to_tensor = transforms.ToTensor()
    bg_paths = {
        "digit": Path(data_root) / "digit" / "bg_digit.png",
        "gelsight": Path(data_root) / "gelsight" / "bg_gelsight.png",
    }
    backgrounds = {}
    for sensor in sensors:
        bg_path = bg_paths[sensor]
        backgrounds[sensor] = to_tensor(Image.open(bg_path).convert("RGB")).unsqueeze(0)
    return backgrounds


def iter_clip_windows(frame_paths, num_frames, stride, window_step):
    window_span = (num_frames - 1) * stride + 1
    if len(frame_paths) < window_span:
        return

    last_start = len(frame_paths) - window_span
    for start_idx in range(0, last_start + 1, window_step):
        yield start_idx, frame_paths[start_idx : start_idx + window_span : stride]


def preprocess_clip(frame_paths, sensor_bg):
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.Normalize(mean, std),
        ]
    )
    frames = [to_tensor(Image.open(path).convert("RGB")) for path in frame_paths]
    clip = torch.stack(frames, dim=0)
    clip = clip - sensor_bg + offset
    clip = torch.clamp(clip, 0.0, 1.0)
    return transform(clip)


def save_metadata(rows, output_dir):
    metadata_path = Path(output_dir) / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sensor", "sequence", "start_frame_index", "frame_paths"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    device = torch.device(args.device)
    random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = data_loader(args.data_root, args.sensors)
    backgrounds = load_sensor_backgrounds(args.data_root, args.sensors)

    model = build_model(args)
    model.to(device)
    model.eval()

    all_cls = []
    all_patch_mean = []
    metadata_rows = []
    total_windows = 0

    with torch.no_grad():
        for sensor, clip_dir, frame_paths in dataset:
            print(f"Processing {sensor}/{clip_dir.name} with {len(frame_paths)} frames")
            sensor_id_tensor = torch.tensor(
                [sensor_name_to_id[sensor]], dtype=torch.long, device=device
            )

            for start_idx, clip_paths in iter_clip_windows(
                frame_paths, args.num_frames, args.stride, args.window_step
            ):
                sensor_bg = backgrounds.get(sensor)
                clip = preprocess_clip(clip_paths, sensor_bg).unsqueeze(0).to(device)
                outputs = model(clip, sensor_id_tensor)

                cls_token = outputs[:, 0, :].squeeze(0).cpu().numpy()
                patch_mean = outputs[:, 6:, :].mean(dim=1).squeeze(0).cpu().numpy()

                all_cls.append(cls_token)
                all_patch_mean.append(patch_mean)
                metadata_rows.append(
                    {
                        "sensor": sensor,
                        "sequence": clip_dir.name,
                        "start_frame_index": start_idx,
                        "frame_paths": "|".join(
                            str(path.relative_to(args.data_root)) for path in clip_paths
                        ),
                    }
                )
                total_windows += 1

    np.save(Path(args.output_dir) / "cls_features.npy", np.stack(all_cls, axis=0))
    np.save(
        Path(args.output_dir) / "patch_mean_features.npy",
        np.stack(all_patch_mean, axis=0),
    )
    save_metadata(metadata_rows, args.output_dir)

    print(f"Processed dataset: {len(dataset)}")
    print(f"Extracted clips: {total_windows}")
    print(f"Saved CLS features to: {Path(args.output_dir) / 'cls_features.npy'}")
    print(f"Saved patch-mean features to: {Path(args.output_dir) / 'patch_mean_features.npy'}")
    print(f"Saved metadata to: {Path(args.output_dir) / 'metadata.csv'}")


if __name__ == "__main__":
    parser = parse_args()
    parser.add_argument("--data_root", type=str, default="mydataset")
    parser.add_argument("--window_step", type=int, default=1,
                        help="Sliding-window step on the sorted frame list")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["digit", "gelsight"]
    )
    args = parser.parse_args()
    main(args)
