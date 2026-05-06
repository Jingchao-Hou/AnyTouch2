#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SPARSH_DIR = THIS_FILE.parents[1]
REPO_ROOT = SPARSH_DIR.parent

# Match the import setup used by the training code.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SPARSH_DIR))

import hydra
import torch
import torch.nn as nn
import torch.utils.data as data
from omegaconf import OmegaConf
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
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find run config: {config_path}")
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def apply_runtime_overrides(cfg, args):
    data_root = str(Path(args.data_root).resolve())
    output_dir = str(Path(args.output_dir).resolve())

    cfg.paths.data_root = data_root
    cfg.paths.tacbench_dir = output_dir
    cfg.paths.output_dir = output_dir
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


def run_single_dataset(cfg, model, dataset_name: str, checkpoint_path: str, device: str):
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


def main():
    args = parse_args()
    cfg = apply_runtime_overrides(load_run_config(args.run_dir), args)

    cfg.test.data.num_workers = args.num_workers
    os.makedirs(cfg.test.path_outputs, exist_ok=True)

    model = build_anytouch_model(cfg, args.checkpoint)
    model.to(args.device)
    model.eval()

    dataset_names = list(cfg.test.data.dataset_name)
    for dataset_name in dataset_names:
        print(f"\nEvaluating {cfg.sensor} on {dataset_name}")
        run_single_dataset(cfg, model, dataset_name, args.checkpoint, args.device)

    if len(dataset_names) > 1:
        tester = TestForceSL(device=args.device, module=model)
        tester.set_test_params(
            task=cfg.experiment_name,
            sensor=cfg.sensor,
            ckpt=extract_epoch_name(args.checkpoint),
            dataset_name="all",
            path_outputs=cfg.test.path_outputs,
            config=cfg,
        )
        tester.get_overall_metrics(build_dataset(cfg, dataset_names[0]), over_all_outputs=True)


if __name__ == "__main__":
    main()
