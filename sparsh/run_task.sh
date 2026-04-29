#!/bin/bash

#SBATCH --job-name=sparsh_force_train_gelsight_anytouch
#SBATCH --partition=gpuv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Running on:"
hostname

echo "GPU status:"
nvidia-smi
### EXPERIMENT: force/gelsight_anytouch, force/digit_anytouch, pose/digit_anytouch, slip/digit_anytouch, slip/gelsight_anytouch
EXPERIMENT=force/gelsight_anytouch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -u "${SCRIPT_DIR}/train_task.py" --config-name=experiment/downstream_task/${EXPERIMENT} paths=default wandb=akash \
    +size=base \
    +load_from_clip=False \
    +old_version=False \
    +ckpt_path='checkpoints/checkpoint-4frames.pth' \
    +input_diff=True \
    +two_frame=False \
    # two_frame=True for checkpoint-2frames.pth, False for checkpoint-4frames.pth
