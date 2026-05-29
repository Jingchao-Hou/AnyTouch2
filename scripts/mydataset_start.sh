CUDA_VISIBLE_DEVICES=0 python3 -u scripts/mydataset_start.py \
    --output_dir log/mydataset_start \
    --data_root mydataset \
    --model_size base \
    --load_path checkpoints/checkpoint-4frames.pth \
    --num_frames 4 \
    --stride 2 \
    --window_step 1 \
    --model anytouch \
    --sensors digit gelsight
