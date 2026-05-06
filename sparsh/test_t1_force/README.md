# Test Sparsh T1 Force Checkpoints

This folder contains a small evaluator for the force task checkpoints saved under `sparsh/log/`.

## What it does

It:

- loads the saved downstream force decoder checkpoint such as `epoch-0051.pth`
- rebuilds the AnyTouch encoder using the same training config saved in the run directory
- runs `sparsh.tactile_ssl.test.TestForceSL`
- saves predictions, metrics, and plots under `sparsh/test_t1_force/outputs/`

## Why test the checkpoint

Training loss only tells you how well the model fits the training/validation loop you used during training.
Testing the checkpoint on held-out force data tells you whether the checkpoint actually generalizes to unseen trajectories and unseen batches.

For this task, the main outputs are:

- `RMSE`: how far the predicted force is from the ground-truth force
- `Correlation`: whether the prediction follows the same trend as the real force in `Fx`, `Fy`, and `Fz`

## Which dataset to use

Use the held-out force dataset split for the same sensor family:

- `digit` checkpoint: test on `digit-force/flat/batch_1`
- `gelsight` checkpoint: test on `gelsight-force/flat/batch_1` and optionally `gelsight-force/flat/batch_2`

That matches the repo configs used for TacBench-style evaluation.

## Example commands

Run these from the repo root after activating the same environment you used for training.

### DIGIT

```bash
python sparsh/test_t1_force/eval_t1_force.py \
  --run-dir sparsh/log/2026.04.30_17-54_digit_t1_force_anytouch_vitbase_1.0 \
  --checkpoint sparsh/log/2026.04.30_17-54_digit_t1_force_anytouch_vitbase_1.0/checkpoints/epoch-0051.pth \
  --data-root /home/jhou-iit.local/AnyTouch2/datasets \
  --dataset-name flat/batch_1 \
  --device cuda:0
```

### GelSight

```bash
python sparsh/test_t1_force/eval_t1_force.py \
  --run-dir sparsh/log/2026.04.30_13-56_gelsight_t1_force_anytouch_vitbase_1.0 \
  --checkpoint sparsh/log/2026.04.30_13-56_gelsight_t1_force_anytouch_vitbase_1.0/checkpoints/epoch-0051.pth \
  --data-root /home/jhou-iit.local/AnyTouch2/datasets \
  --dataset-name flat/batch_1 \
  --dataset-name flat/batch_2 \
  --device cuda:0
```

## Outputs

The evaluator writes files under:

```bash
sparsh/test_t1_force/outputs/
```

You should see:

- `*_predictions.npy`
- `*_metrics.npy`
- `*_correlation.png`
- `*_XYZerror.png`
- `*_ConeError.png`
