#!/bin/bash
#SBATCH --job-name=sparsh_force
#SBATCH --partition=gpua
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/fastwork/jhou/AnyTouch2/logs/%x_%j.out
#SBATCH --error=/fastwork/jhou/AnyTouch2/logs/%x_%j.err

echo "Running on:"
hostname

echo "GPU status:"
nvidia-smi

echo "Activating conda environment..."

source ~/.bashrc
conda activate anytouch2

echo "Python path:"
which python

echo "Working directory setup..."

cd /fastwork/jhou/AnyTouch2/sparsh

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
