#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from datetime import datetime
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SPARSH_DIR = THIS_FILE.parents[1]
REPO_ROOT = SPARSH_DIR.parent

# Match the import setup used by the training code.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SPARSH_DIR))

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoConfig

from model.tactile_mae import TactileVideoMAE
from sparsh.tactile_ssl.test import TestForceSL


_SENSOR_TYPE_MAP = {"digit": 1, "gelsight": 3}


class _AnyTouchEncoderWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, sensor_type: int):
        super().__init__()
        self.base_model = base_model
        self._sensor_type = sensor_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensor = torch.full(
            (x.shape[0],), self._sensor_type, dtype=torch.long, device=x.device
        )
        return self.base_model(x, sensor_type=sensor, probe=True)


def load_model_from_multi_clip(ckpt, model):
    new_ckpt = {}
    for key, item in ckpt.items():
        if (
            "touch_mae_model" in key
            and "decoder" not in key
            and "mask_token" not in key
        ):
            new_ckpt[key.replace("touch_mae_model.", "")] = copy.deepcopy(item)
    model.load_state_dict(new_ckpt, strict=True)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Sparsh t1 force checkpoints saved under sparsh/log/."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory, e.g. sparsh/log/2026.04.30_17-54_digit_t1_force_anytouch_vitbase_1.0",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Task checkpoint to evaluate, e.g. .../checkpoints/epoch-0051.pth",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root directory, e.g. /home/.../AnyTouch2/datasets",
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        dest="dataset_names",
        help="Dataset split(s) to test. Repeat this flag for multiple splits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override test batch size. Defaults to cfg.test.data.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers for evaluation.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device string, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SPARSH_DIR / "test_t1_force" / "outputs"),
        help="Directory for predictions, plots, and metrics.",
    )
    return parser.parse_args()


def extract_epoch_name(checkpoint_path: str) -> str:
    match = re.search(r"epoch-(\d+)\.(?:pth|ckpt)$", os.path.basename(checkpoint_path))
    if not match:
        raise ValueError(
            "Checkpoint name must look like epoch-XXXX.pth or epoch-XXXX.ckpt. "
            f"Got: {checkpoint_path}"
        )
    return f"epoch-{int(match.group(1)):04d}.pth"


def load_run_config(run_dir: str):
    run_dir = str(Path(run_dir).resolve())
    config_candidates = [
        os.path.join(run_dir, "config.yaml"),
        os.path.join(run_dir, ".hydra", "config.yaml"),
    ]
    config_path = next((p for p in config_candidates if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError(
            "Could not find a run config. Checked:\n" + "\n".join(config_candidates)
        )
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def apply_runtime_overrides(cfg, args):
    data_root = str(Path(args.data_root).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    run_output_dir = os.path.join(
        output_dir,
        f"{cfg.task_name}_{cfg.sensor}_{timestamp}",
    )

    cfg.paths.data_root = data_root
    cfg.paths.tacbench_dir = run_output_dir
    cfg.paths.output_dir = run_output_dir
    cfg.paths.work_dir = str(REPO_ROOT)
    if not os.path.isabs(cfg.ckpt_path):
        cfg.ckpt_path = str((REPO_ROOT / cfg.ckpt_path).resolve())
    cfg.data.dataset.config.path_dataset = os.path.join(
        data_root,
        "digit-force" if cfg.data.sensor == "digit" else "gelsight-force",
    )

    if args.dataset_names:
        cfg.test.data.dataset_name = list(args.dataset_names)

    if args.batch_size is not None:
        cfg.test.data.batch_size = args.batch_size

    cfg.test.path_outputs = os.path.join(run_output_dir, f"{cfg.task_name}_{cfg.sensor}")
    return cfg


def build_anytouch_model(cfg, checkpoint_task: str):
    num_frames = cfg.data.dataset.config.num_frames
    frame_stride = cfg.data.dataset.config.frame_stride
    size = cfg.get("size", cfg.ssl_model_size)
    if size != "base":
        raise ValueError(f"Unsupported AnyTouch size for this evaluator: {size}")

    clip_config_path = REPO_ROOT / "CLIP-B-16"
    config = AutoConfig.from_pretrained(str(clip_config_path))
    mae_args = argparse.Namespace(mask_ratio=0.0, stride=frame_stride)
    base_encoder = TactileVideoMAE(mae_args, config, num_frames, tube_size=1)
    base_encoder = load_model_from_multi_clip(
        torch.load(cfg.ckpt_path, map_location="cpu"), base_encoder
    )

    sensor_int = _SENSOR_TYPE_MAP[cfg.data.sensor]
    model_encoder = _AnyTouchEncoderWrapper(base_encoder, sensor_int)

    cfg.task.model_task.embed_dim = size
    cfg.task.checkpoint_encoder = None
    cfg.task.checkpoint_task = checkpoint_task
    del cfg.task.model_encoder

    model = hydra.utils.instantiate(cfg.task, model_encoder=model_encoder)
    model.model_encoder.requires_grad_(False)
    model.model_encoder.eval()
    return model


def build_dataset(cfg, dataset_name: str):
    return hydra.utils.instantiate(
        cfg.data.dataset,
        dataset_name=dataset_name,
        is_train=False,
    )


def load_scaled_outputs(tester, dataset):
    outputs = np.load(
        f"{tester.path_outputs}/{tester.epoch}_predictions.npy", allow_pickle=True
    ).item()
    scale = np.asarray(dataset.max_abs_forceXYZ, dtype=np.float32)
    forces_gt = outputs["forces_gt"] * scale
    forces_pred = outputs["forces_pred"] * scale
    return forces_gt, forces_pred


def compute_force_metrics(forces_gt: np.ndarray, forces_pred: np.ndarray):
    per_sample_rmse = np.sqrt((forces_gt - forces_pred) ** 2).mean(axis=1)
    rmse = float(per_sample_rmse.mean())
    rmse_std = float(per_sample_rmse.std())
    corr = np.array(
        [
            np.corrcoef(forces_gt[:, i], forces_pred[:, i])[0, 1]
            for i in range(forces_gt.shape[1])
        ],
        dtype=np.float32,
    )
    sem = rmse_std / np.sqrt(forces_gt.shape[0])
    ci95 = 1.96 * sem
    return {
        "rmse": rmse,
        "rmse_std": rmse_std,
        "corr": corr,
        "sem": float(sem),
        "ci95": float(ci95),
        "n_samples": int(forces_gt.shape[0]),
    }


def make_force_magnitude_bar_plot(
    forces_gt: np.ndarray,
    forces_pred: np.ndarray,
    dataset_name: str,
    output_prefix: str,
    n_bins: int = 8,
):
    force_magnitudes = np.linalg.norm(forces_gt, axis=1)
    sample_rmse = np.sqrt(((forces_gt - forces_pred) ** 2).mean(axis=1))

    max_mag = float(force_magnitudes.max())
    bin_edges = np.linspace(0.0, max_mag, n_bins + 1)
    bin_indices = np.digitize(force_magnitudes, bin_edges[1:-1], right=False)

    avg_rmse = []
    counts = []
    labels = []
    centers = []
    for i in range(n_bins):
        mask = bin_indices == i
        left = bin_edges[i]
        right = bin_edges[i + 1]
        labels.append(f"{left:.2f}-{right:.2f}")
        centers.append((left + right) / 2.0)
        counts.append(int(mask.sum()))
        avg_rmse.append(float(sample_rmse[mask].mean()) if mask.any() else np.nan)

    stats = {
        "dataset_name": dataset_name,
        "bin_edges": bin_edges,
        "bin_centers": np.asarray(centers, dtype=np.float32),
        "bin_labels": np.asarray(labels),
        "avg_rmse": np.asarray(avg_rmse, dtype=np.float32),
        "avg_rmse_mn": np.asarray(avg_rmse, dtype=np.float32) * 1000.0,
        "counts": np.asarray(counts, dtype=np.int32),
    }
    np.save(f"{output_prefix}_force_magnitude_stats.npy", stats)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_bins)
    bar_values = np.nan_to_num(stats["avg_rmse_mn"], nan=0.0)
    ax.bar(x, bar_values, color="#4C78A8", edgecolor="black", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{center:.2f}" for center in stats["bin_centers"]])
    ax.set_xlabel("Force Magnitude Bin Center (N)")
    ax.set_ylabel("Average RMSE (mN)")
    ax.set_title(f"Average RMSE by Force Magnitude: {dataset_name}")
    ax.grid(axis="y", alpha=0.3)
    for idx, (value_mn, count, label) in enumerate(
        zip(stats["avg_rmse_mn"], stats["counts"], stats["bin_labels"])
    ):
        if not np.isnan(value_mn):
            ax.text(
                idx,
                value_mn,
                f"{label} N\n{value_mn:.1f} mN\nn={count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_force_magnitude_rmse.png", dpi=200)

    img_buf = Path(f"{output_prefix}_force_magnitude_rmse.png")
    img = Image.open(img_buf)
    plt.close(fig)
    return stats, img


def init_eval_wandb(cfg, checkpoint_path: str):
    run_name = (
        f"{cfg.sensor}_{cfg.task_name}_eval_"
        f"{Path(checkpoint_path).stem}"
    )
    return wandb.init(
        project="Eval-f1-force",
        group="t1_force_eval",
        name=run_name,
        dir=cfg.paths.output_dir,
        config={
            "sensor": cfg.sensor,
            "task_name": cfg.task_name,
            "experiment_name": cfg.experiment_name,
            "checkpoint_task": checkpoint_path,
            "checkpoint_encoder": cfg.ckpt_path,
            "datasets": list(cfg.test.data.dataset_name),
        },
    )


def run_single_dataset(
    cfg,
    model,
    dataset_name: str,
    checkpoint_path: str,
    device: str,
    wandb_run=None,
):
    dataset = build_dataset(cfg, dataset_name)
    dataloader = data.DataLoader(
        dataset,
        batch_size=cfg.test.data.batch_size,
        num_workers=int(cfg.test.data.num_workers),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    tester = TestForceSL(device=device, module=model)
    tester.set_test_params(
        task=cfg.experiment_name,
        sensor=cfg.sensor,
        ckpt=extract_epoch_name(checkpoint_path),
        dataset_name=dataset_name,
        path_outputs=cfg.test.path_outputs,
        config=cfg,
    )
    tester.run_model(dataset, dataloader)
    tester.get_overall_metrics(dataset)
    tester.make_plots(dataset)

    forces_gt, forces_pred = load_scaled_outputs(tester, dataset)
    output_prefix = f"{tester.path_outputs}/{tester.epoch}"
    mag_stats, mag_plot = make_force_magnitude_bar_plot(
        forces_gt,
        forces_pred,
        dataset_name=tester.dataset_name,
        output_prefix=output_prefix,
    )

    if wandb_run is not None:
        metrics_path = (
            f"{tester.path_output_model}/{tester.epoch}_{tester.dataset_name}_metrics.npy"
        )
        metrics = np.load(metrics_path, allow_pickle=True).item()
        wandb_payload = {
            f"{tester.dataset_name}/rmse": float(metrics["rmse"]),
            f"{tester.dataset_name}/rmse_std": float(metrics["rmse_std"]),
            f"{tester.dataset_name}/corr_fx": float(metrics["corr"][0]),
            f"{tester.dataset_name}/corr_fy": float(metrics["corr"][1]),
            f"{tester.dataset_name}/corr_fz": float(metrics["corr"][2]),
            f"{tester.dataset_name}/n_samples": int(metrics["n_samples"]),
            f"{tester.dataset_name}/force_magnitude_rmse": wandb.Image(mag_plot),
        }
        wandb_run.log(wandb_payload)

    return {
        "forces_gt": forces_gt,
        "forces_pred": forces_pred,
        "metrics": metrics if wandb_run is not None else None,
    }


def main():
    args = parse_args()
    cfg = apply_runtime_overrides(load_run_config(args.run_dir), args)

    cfg.test.data.num_workers = args.num_workers
    os.makedirs(cfg.test.path_outputs, exist_ok=True)

    model = build_anytouch_model(cfg, args.checkpoint)
    model.to(args.device)
    model.eval()
    wandb_run = init_eval_wandb(cfg, args.checkpoint)

    dataset_names = list(cfg.test.data.dataset_name)
    collected_outputs = []
    for dataset_name in dataset_names:
        print(f"\nEvaluating {cfg.sensor} on {dataset_name}")
        result = run_single_dataset(
            cfg, model, dataset_name, args.checkpoint, args.device, wandb_run=wandb_run
        )
        collected_outputs.append(result)

    if len(dataset_names) > 1:
        all_forces_gt = np.vstack([item["forces_gt"] for item in collected_outputs])
        all_forces_pred = np.vstack([item["forces_pred"] for item in collected_outputs])
        metrics = compute_force_metrics(all_forces_gt, all_forces_pred)
        task_output_dir = os.path.join(cfg.test.path_outputs, cfg.experiment_name)
        os.makedirs(task_output_dir, exist_ok=True)
        epoch_num = int(extract_epoch_name(args.checkpoint).split("-")[-1].split(".")[0])
        np.save(
            os.path.join(task_output_dir, f"{epoch_num}_metrics.npy"),
            metrics,
        )
        print("Metrics for all outputs:")
        print(f"RMSE: {metrics['rmse']} ± {metrics['rmse_std']} N")
        print(f"Correlation: {metrics['corr']}")
        print(f"Total samples: {metrics['n_samples']}")
        wandb_run.log(
            {
                "all/rmse": float(metrics["rmse"]),
                "all/rmse_std": float(metrics["rmse_std"]),
                "all/corr_fx": float(metrics["corr"][0]),
                "all/corr_fy": float(metrics["corr"][1]),
                "all/corr_fz": float(metrics["corr"][2]),
                "all/n_samples": int(metrics["n_samples"]),
            }
        )
    wandb_run.finish()


if __name__ == "__main__":
    main()
