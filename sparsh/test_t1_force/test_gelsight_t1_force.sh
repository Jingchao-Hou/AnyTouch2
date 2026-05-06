#!/bin/bash


echo "Running on:"
hostname

echo "GPU status:"
nvidia-smi


source ~/.bashrc
conda activate anytouch2



python sparsh/test_t1_force/eval_t1_force.py \
  --run-dir sparsh/log/2026.04.30_13-56_gelsight_t1_force_anytouch_vitbase_1.0 \
  --checkpoint sparsh/log/2026.04.30_13-56_gelsight_t1_force_anytouch_vitbase_1.0/checkpoints/epoch-0051.pth \
  --data-root set -euo pipefail \
  --dataset-name flat/batch_1 \
  --dataset-name flat/batch_2 \
  --device cuda:0
